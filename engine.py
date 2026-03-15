"""Taichi AD engine for 2D/3D topology optimisation.

Uses Taichi automatic differentiation for filter/projection chain,
and adjoint method for FEA sensitivity (PCG solve is not differentiated).
"""
import numpy as np


def make_engine_class():
    """Return the TopOptEngine class.  Must be called after ``ti.init()``."""
    import taichi as ti
    from mesh import MeshData

    @ti.data_oriented
    class TopOptEngine:

        def __init__(self, mesh: MeshData, KE_np: np.ndarray,
                     E0: float, Emin: float, penal: float,
                     eta: float, beta: float):
            self.mesh = mesh
            self.n_elem = mesh.n_elem
            self.n_active = mesh.n_active
            self.n_dof = mesh.n_dof
            self.dpe = mesh.dpe
            self.E0 = float(E0)
            self.Emin = float(Emin)

            self.penal_val = ti.field(dtype=ti.f64, shape=())
            self.beta_val = ti.field(dtype=ti.f64, shape=())
            self.eta_val = ti.field(dtype=ti.f64, shape=())
            self.inv_n_active = ti.field(dtype=ti.f64, shape=())
            self.penal_val[None] = float(penal)
            self.beta_val[None] = float(beta)
            self.eta_val[None] = float(eta)
            self.inv_n_active[None] = 1.0 / max(1, self.n_active)

            self.active = ti.field(dtype=ti.i32, shape=self.n_elem)
            self.active_f = ti.field(dtype=ti.f64, shape=self.n_elem)
            self.Hs_f = ti.field(dtype=ti.f64, shape=self.n_elem)
            self.row_ptr = ti.field(dtype=ti.i32, shape=self.n_elem + 1)
            self.col_idx = ti.field(dtype=ti.i32, shape=mesh.col_idx.size)
            self.wt = ti.field(dtype=ti.f64, shape=mesh.weights.size)
            self.edof = ti.field(dtype=ti.i32, shape=(self.n_active, self.dpe))
            self.elem_id = ti.field(dtype=ti.i32, shape=self.n_active)
            self.KE = ti.field(dtype=ti.f64, shape=(self.dpe, self.dpe))

            self.x = ti.field(dtype=ti.f64, shape=self.n_elem, needs_grad=True)
            self.x_tilde = ti.field(dtype=ti.f64, shape=self.n_elem, needs_grad=True)
            self.x_phys = ti.field(dtype=ti.f64, shape=self.n_elem, needs_grad=True)
            self.loss = ti.field(dtype=ti.f64, shape=(), needs_grad=True)

            self.u = ti.field(dtype=ti.f64, shape=self.n_dof)
            self.b = ti.field(dtype=ti.f64, shape=self.n_dof)
            self.r = ti.field(dtype=ti.f64, shape=self.n_dof)
            self.z = ti.field(dtype=ti.f64, shape=self.n_dof)
            self.p_vec = ti.field(dtype=ti.f64, shape=self.n_dof)
            self.Ap = ti.field(dtype=ti.f64, shape=self.n_dof)
            self.diag_f = ti.field(dtype=ti.f64, shape=self.n_dof)
            self.free = ti.field(dtype=ti.i32, shape=self.n_dof)

            self.ce = ti.field(dtype=ti.f64, shape=self.n_active)
            self.acc = ti.field(dtype=ti.f64, shape=())

            self.active.from_numpy(mesh.active_mask.astype(np.int32))
            self.active_f.from_numpy(mesh.active_mask.astype(np.float64))
            self.Hs_f.from_numpy(mesh.Hs.astype(np.float64))
            self.row_ptr.from_numpy(mesh.row_ptr)
            self.col_idx.from_numpy(mesh.col_idx)
            self.wt.from_numpy(mesh.weights)
            self.edof.from_numpy(mesh.edof_active.astype(np.int32))
            self.elem_id.from_numpy(mesh.active_ids.astype(np.int32))
            self.free.from_numpy(mesh.free_mask.astype(np.int32))
            self.b.from_numpy(mesh.F.astype(np.float64))
            for i in range(self.dpe):
                for j in range(self.dpe):
                    self.KE[i, j] = float(KE_np[i, j])

        def set_params(self, penal: float, beta: float):
            self.penal_val[None] = float(penal)
            self.beta_val[None] = float(beta)

        def set_x(self, x_np: np.ndarray):
            self.x.from_numpy(x_np.astype(np.float64))

        # ---- forward (AD-tracked) ------------------------------------
        @ti.kernel
        def _filter_fwd(self):
            for i in range(self.n_elem):
                s = 0.0
                for p in range(self.row_ptr[i], self.row_ptr[i + 1]):
                    s += self.wt[p] * self.x[self.col_idx[p]]
                self.x_tilde[i] = self.active_f[i] * s / self.Hs_f[i]

        @ti.kernel
        def _proj_fwd(self):
            for i in range(self.n_elem):
                beta = self.beta_val[None]
                eta = self.eta_val[None]
                den = ti.tanh(beta * eta) + ti.tanh(beta * (1.0 - eta))
                raw = (ti.tanh(beta * eta)
                       + ti.tanh(beta * (self.x_tilde[i] - eta))) / den
                self.x_phys[i] = self.active_f[i] * raw

        def forward(self):
            self._filter_fwd()
            self._proj_fwd()

        # ---- PCG kernels (unchanged) ---------------------------------
        @ti.kernel
        def _build_diag(self):
            for i in range(self.n_dof):
                self.diag_f[i] = 1e-12
            for ae in range(self.n_active):
                eid = self.elem_id[ae]
                dens = ti.max(self.x_phys[eid], 1e-9)
                Ee = self.Emin + ti.pow(dens, self.penal_val[None]) * (self.E0 - self.Emin)
                for i in ti.static(range(self.dpe)):
                    ti.atomic_add(self.diag_f[self.edof[ae, i]], Ee * self.KE[i, i])
            for i in range(self.n_dof):
                if self.free[i] == 0:
                    self.diag_f[i] = 1.0
                else:
                    self.diag_f[i] = 1.0 / ti.max(self.diag_f[i], 1e-12)

        @ti.kernel
        def _matvec_u(self):
            for i in range(self.n_dof):
                self.Ap[i] = 0.0
            for ae in range(self.n_active):
                eid = self.elem_id[ae]
                dens = ti.max(self.x_phys[eid], 1e-9)
                Ee = self.Emin + ti.pow(dens, self.penal_val[None]) * (self.E0 - self.Emin)
                for i in ti.static(range(self.dpe)):
                    dofi = self.edof[ae, i]
                    a = 0.0
                    for j in ti.static(range(self.dpe)):
                        a += self.KE[i, j] * self.u[self.edof[ae, j]]
                    ti.atomic_add(self.Ap[dofi], Ee * a)

        @ti.kernel
        def _matvec_p(self):
            for i in range(self.n_dof):
                self.Ap[i] = 0.0
            for ae in range(self.n_active):
                eid = self.elem_id[ae]
                dens = ti.max(self.x_phys[eid], 1e-9)
                Ee = self.Emin + ti.pow(dens, self.penal_val[None]) * (self.E0 - self.Emin)
                for i in ti.static(range(self.dpe)):
                    dofi = self.edof[ae, i]
                    a = 0.0
                    for j in ti.static(range(self.dpe)):
                        a += self.KE[i, j] * self.p_vec[self.edof[ae, j]]
                    ti.atomic_add(self.Ap[dofi], Ee * a)

        @ti.kernel
        def _zero_u(self):
            for i in range(self.n_dof):
                self.u[i] = 0.0

        @ti.kernel
        def _residual_init(self):
            for i in range(self.n_dof):
                self.r[i] = (self.b[i] - self.Ap[i]) if self.free[i] == 1 else 0.0

        @ti.kernel
        def _precond(self):
            for i in range(self.n_dof):
                self.z[i] = (self.diag_f[i] * self.r[i]) if self.free[i] == 1 else 0.0

        @ti.kernel
        def _copy_p_from_z(self):
            for i in range(self.n_dof):
                self.p_vec[i] = self.z[i] if self.free[i] == 1 else 0.0

        @ti.kernel
        def _dot_rz(self):
            self.acc[None] = 0.0
            for i in range(self.n_dof):
                if self.free[i] == 1:
                    self.acc[None] += self.r[i] * self.z[i]

        @ti.kernel
        def _dot_rr(self):
            self.acc[None] = 0.0
            for i in range(self.n_dof):
                if self.free[i] == 1:
                    self.acc[None] += self.r[i] * self.r[i]

        @ti.kernel
        def _dot_pAp(self):
            self.acc[None] = 0.0
            for i in range(self.n_dof):
                if self.free[i] == 1:
                    self.acc[None] += self.p_vec[i] * self.Ap[i]

        @ti.kernel
        def _update_u(self, alpha: ti.f64):
            for i in range(self.n_dof):
                if self.free[i] == 1:
                    self.u[i] += alpha * self.p_vec[i]

        @ti.kernel
        def _update_r(self, neg_alpha: ti.f64):
            for i in range(self.n_dof):
                if self.free[i] == 1:
                    self.r[i] += neg_alpha * self.Ap[i]
                else:
                    self.r[i] = 0.0

        @ti.kernel
        def _update_p(self, beta_cg: ti.f64):
            for i in range(self.n_dof):
                if self.free[i] == 1:
                    self.p_vec[i] = self.z[i] + beta_cg * self.p_vec[i]
                else:
                    self.p_vec[i] = 0.0

        # ---- PCG solve (unchanged) -----------------------------------
        def solve_pcg(self, max_iter: int, tol: float):
            self._zero_u()
            self._build_diag()
            self._matvec_u()
            self._residual_init()
            self._precond()
            self._copy_p_from_z()

            self._dot_rz(); rz_old = float(self.acc[None])
            self._dot_rr(); rnorm0 = max(float(self.acc[None]) ** 0.5, 1e-30)
            rnorm = rnorm0
            it_used = 0

            for k in range(max_iter):
                self._matvec_p()
                self._dot_pAp(); pAp = float(self.acc[None])
                if abs(pAp) < 1e-30:
                    break
                alpha = rz_old / pAp
                self._update_u(alpha)
                self._update_r(-alpha)
                self._dot_rr(); rnorm = float(self.acc[None]) ** 0.5
                it_used = k + 1
                if rnorm <= tol * rnorm0 or rnorm <= tol:
                    break
                self._precond()
                self._dot_rz(); rz_new = float(self.acc[None])
                beta_cg = rz_new / max(rz_old, 1e-30)
                self._update_p(beta_cg)
                rz_old = rz_new

            return self.u.to_numpy(), it_used, rnorm

        # ---- adjoint sensitivity -------------------------------------
        @ti.kernel
        def _compute_ce(self):
            for ae in range(self.n_active):
                c = 0.0
                for i in ti.static(range(self.dpe)):
                    for j in ti.static(range(self.dpe)):
                        c += self.KE[i, j] * self.u[self.edof[ae, i]] \
                             * self.u[self.edof[ae, j]]
                self.ce[ae] = c

        @ti.kernel
        def _seed_xphys_grad_compliance(self):
            for i in range(self.n_elem):
                self.x_phys.grad[i] = 0.0
            for ae in range(self.n_active):
                eid = self.elem_id[ae]
                dens = ti.max(self.x_phys[eid], 1e-9)
                p = self.penal_val[None]
                self.x_phys.grad[eid] = (
                    -p * ti.pow(dens, p - 1.0) * (self.E0 - self.Emin) * self.ce[ae]
                )

        @ti.kernel
        def _seed_xphys_grad_volume(self):
            for i in range(self.n_elem):
                self.x_phys.grad[i] = 0.0
            for ae in range(self.n_active):
                eid = self.elem_id[ae]
                self.x_phys.grad[eid] = self.inv_n_active[None]

        @ti.kernel
        def _zero_grads(self):
            for i in range(self.n_elem):
                self.x.grad[i] = 0.0
                self.x_tilde.grad[i] = 0.0

        @ti.kernel
        def _compliance_val(self):
            self.acc[None] = 0.0
            for i in range(self.n_dof):
                if self.free[i] == 1:
                    self.acc[None] += self.b[i] * self.u[i]

        @ti.kernel
        def _volume_val(self):
            self.acc[None] = 0.0
            for i in range(self.n_elem):
                if self.active[i] == 1:
                    self.acc[None] += self.x_phys[i] * self.inv_n_active[None]

        def _backprop_to_x(self):
            """Run AD backward: x_phys.grad → x_tilde.grad → x.grad.

            Re-executes forward kernels so Taichi's tape captures the
            correct primal values, then calls .grad() in reverse order.
            """
            self._filter_fwd()
            self._proj_fwd()
            self._proj_fwd.grad()
            self._filter_fwd.grad()

        def compute_compliance_grad(self):
            """Return (dc_numpy, compliance_float)."""
            self._compute_ce()
            self._compliance_val()
            compliance = float(self.acc[None])
            self._zero_grads()
            self._seed_xphys_grad_compliance()
            self._backprop_to_x()
            return self.x.grad.to_numpy(), compliance

        def compute_volume_grad(self):
            """Return (dv_numpy, volume_float)."""
            self._volume_val()
            volume = float(self.acc[None])
            self._zero_grads()
            self._seed_xphys_grad_volume()
            self._backprop_to_x()
            return self.x.grad.to_numpy(), volume

        def get_x_phys(self):
            return self.x_phys.to_numpy()

        def get_u(self):
            return self.u.to_numpy()

    return TopOptEngine
