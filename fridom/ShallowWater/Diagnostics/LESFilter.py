from fridom.ShallowWater.ModelSettings import ModelSettings
from fridom.ShallowWater.Grid import Grid
from fridom.ShallowWater.State import State
from fridom.Framework.FieldVariable import FieldVariable


class LESFilter:
    def __init__(self, mset:ModelSettings, grid:Grid, kmax: float):
        """
        Constructor for LESFilter class which is used to filter the variables
        of the shallow water model and to compute subgrid energy terms.

        Arguments:
            mset (ModelSettings): Model settings object
            grid (Grid)         : Grid object
            kmax (float)        : Filter width for LES filter (in wave numbers)
        """
        self.mset = mset
        self.grid = grid
        self.kmax = kmax

    def filter_field(self, field: FieldVariable):
        """
        Filters the given field variable using the LES filter.

        Arguments:
            field (FieldVariable): Field variable to be filtered
        """
        # transform to spectral space if not already
        if field.is_spectral:
            f_hat = field.copy()
        else:
            f_hat = field.fft()
        
        # make wavenumber mask
        kx, ky = self.grid.K
        k = (kx**2 + ky**2)**0.5
        mask = (k <= self.kmax).astype(int)

        # apply mask
        f_hat *= mask

        # transform back to physical space if it was originally in physical space
        if not field.is_spectral:
            f_hat = f_hat.fft()

        return f_hat


    def filter_state(self, state: State):
        """
        Returns a new State object with the filtered state variables.

        Arguments:
            state (State): State object to be filtered

        Returns:
            filtered_state (State): Filtered state object
        """
        filtered_z = State(self.mset, self.grid, is_spectral=state.is_spectral)

        filtered_z.h = self.filter_field(state.h)
        filtered_z.u = self.filter_field(state.u)
        filtered_z.v = self.filter_field(state.v)
        return filtered_z

    def differentiate(self, field: FieldVariable, axis: int, order: int = 1):
        """
        Computes the derivative of the given field variable in spectral space.

        Arguments:
            field (FieldVariable): Field variable to be differentiated
            axis (int)           : Axis along which to differentiate
            order (int)          : Order of differentiation
        """
        # transform to spectral space if not already
        if field.is_spectral:
            f_hat = field.copy()
        else:
            f_hat = field.fft()
        
        # get wavenumber array
        k = self.grid.K[axis].copy()

        # set k to 1 where k is 0 to avoid division by 0
        k_zero = (k == 0)
        k[k_zero] = 1

        # compute derivative
        f_hat *= (1j * k)**order

        # set f_hat to 0 where k is 0
        f_hat[k_zero] = 0

        # transform back to physical space if it was originally in physical space
        if not field.is_spectral:
            f_hat = f_hat.fft()

        return f_hat

    def leonard_correction(self, zf: State):
        """
        Computes the Leonard correction term for the given state object.

        Arguments:
            zf (State): Full state object (not filtered)

        Returns:
            zl (State): State object containing the Leonard correction term
        """
        z = self.filter_state(zf)
        # transform to physical space if not already
        if z.is_spectral:
            z = z.fft()

        # compute derivatives
        dudx = self.differentiate(z.u, 0)
        dudy = self.differentiate(z.u, 1)
        dvdx = self.differentiate(z.v, 0)
        dvdy = self.differentiate(z.v, 1)

        # calc non-linear terms
        u_grad_u = z.u * dudx + z.v * dudy
        u_grad_v = z.u * dvdx + z.v * dvdy
        duhdx = self.differentiate(z.u*z.h, 0)
        dvhdy = self.differentiate(z.v*z.h, 1)

        zl = State(self.mset, self.grid, is_spectral=False)
        zl.u[:] = self.filter_field(u_grad_u) - u_grad_u
        zl.v[:] = self.filter_field(u_grad_v) - u_grad_v
        zl.h[:] = self.filter_field(duhdx + dvhdy) - (duhdx + dvhdy)

        return zl

    def clark_correction(self, zf:State):
        """
        Computes the Clark correction term for the given state object.

        Arguments:
            zf (State): Full state object (not filtered)

        Returns:
            zc (State): State object containing the Clark correction term
        """
        # transform to physical space if not already
        if zf.is_spectral:
            zf = zf.fft()
        else:
            zf = zf.copy()

        # filter and residual
        z = self.filter_state(zf)
        zr = zf - z

        # compute derivatives
        dudx = self.differentiate(z.u, 0)
        dudy = self.differentiate(z.u, 1)
        dvdx = self.differentiate(z.v, 0)
        dvdy = self.differentiate(z.v, 1)

        durdx = self.differentiate(zr.u, 0)
        durdy = self.differentiate(zr.u, 1)
        dvrdx = self.differentiate(zr.v, 0)
        dvrdy = self.differentiate(zr.v, 1)

        # calc non-linear terms
        u_grad_ur = z.u * durdx + z.v * durdy
        ur_grad_u = zr.u * dudx + zr.v * dudy
        u_grad_vr = z.u * dvrdx + z.v * dvrdy
        ur_grad_v = zr.u * dvdx + zr.v * dvdy
        durhdx = self.differentiate(zr.u*z.h, 0)
        duhrdx = self.differentiate(z.u*zr.h, 0)
        dvrhdy = self.differentiate(zr.v*z.h, 1)
        dvhrdy = self.differentiate(z.v*zr.h, 1)

        zc = State(self.mset, self.grid, is_spectral=False)
        zc.u[:] = self.filter_field(u_grad_ur + ur_grad_u) 
        zc.v[:] = self.filter_field(u_grad_vr + ur_grad_v)
        zc.h[:] = self.filter_field(durhdx + duhrdx + dvrhdy + dvhrdy)

        return zc

    def reynolds_correction(self, zf:State):
        """
        Computes the Reynolds correction term for the given state object.

        Arguments:
            zf (State): Full state object (not filtered)

        Returns:
            zr (State): State object containing the Reynolds correction term
        """
        # transform to physical space if not already
        if zf.is_spectral:
            zf = zf.fft()
        else:
            zf = zf.copy()

        # residual
        zr = zf - self.filter_state(zf)

        # compute derivatives
        durdx = self.differentiate(zr.u, 0)
        durdy = self.differentiate(zr.u, 1)
        dvrdx = self.differentiate(zr.v, 0)
        dvrdy = self.differentiate(zr.v, 1)

        # calc non-linear terms
        ur_grad_ur = zr.u * durdx + zr.v * durdy
        ur_grad_vr = zr.u * dvrdx + zr.v * dvrdy
        durhrdx = self.differentiate(zr.u*zr.h, 0)
        dvrhrdy = self.differentiate(zr.v*zr.h, 1)

        zr = State(self.mset, self.grid, is_spectral=False)
        zr.u[:] = self.filter_field(ur_grad_ur)
        zr.v[:] = self.filter_field(ur_grad_vr)
        zr.h[:] = self.filter_field(durhrdx + dvrhrdy)

        return zr

    def e_kin_corr(self, z_corr:State, z:State):
        """
        Computes the kinetic energy correction term for the given state tendency.

        Arguments:
            z_corr (State): State object containing the correction term
            z (State)     : State object containing the full (averaged) state

        Returns:
            e_kin_corr (FieldVariable): Field containing the kinetic energy correction term
        """
        C_H = z_corr.h
        C_U = z_corr.u
        C_V = z_corr.v

        H = self.mset.csqr + self.mset.Ro * z.h

        # ekin contribution
        ek = FieldVariable(self.mset, self.grid, is_spectral=False,
                            name='Transfer to unresolved Ekin')
        ek[:] = -self.mset.Ro ** 3 * (
            (z.u**2 + z.v**2) * 0.5 * C_H + H * (z.u * C_U + z.v * C_V)
        )
        return ek

    def e_pot_corr(self, z_corr:State, z:State):
        """
        Computes the potential energy correction term for the given state tendency.

        Arguments:
            z_corr (State): State object containing the correction term
            z (State)     : State object containing the full (averaged) state

        Returns:
            e_pot_corr (FieldVariable): Field containing the potential energy correction term
        """
        C_H = z_corr.h

        H = self.mset.csqr + self.mset.Ro * z.h

        # epot contribution
        ep = FieldVariable(self.mset, self.grid, is_spectral=False,
                            name='Transfer to unresolved Epot')
        ep[:] = -self.mset.Ro * (
            H * C_H 
        )
        return ep

    def leonard_ekin(self, zf:State):
        """
        Computes the Leonard kinetic energy term for the given state object.

        Arguments:
            zf (State): Full state object (not filtered)

        Returns:
            lek (FieldVariable): Field containing the Leonard kinetic energy term
        """
        # transform to physical space if not already
        if zf.is_spectral:
            zf = zf.fft()

        lek = self.e_kin_corr(self.leonard_correction(zf),
                              self.filter_state(zf))
        return lek

    def leonard_epot(self, zf:State):
        """
        Computes the Leonard potential energy term for the given state object.

        Arguments:
            zf (State): Full state object (not filtered)

        Returns:
            lep (FieldVariable): Field containing the Leonard potential energy term
        """
        # transform to physical space if not already
        if zf.is_spectral:
            zf = zf.fft()

        lep = self.e_pot_corr(self.leonard_correction(zf), 
                              self.filter_state(zf))
            
        return lep

    def leonard_etot(self, zf:State):
        """
        Computes the Leonard total energy term for the given state object.

        Arguments:
            zf (State): Full state object (not filtered)

        Returns:
            let (FieldVariable): Field containing the Leonard total energy term
        """
        let = self.leonard_ekin(zf) + self.leonard_epot(zf)
        let.name = "Transfer to unresolved Etot"

        return let

    def clark_ekin(self, zf:State):
        """
        Computes the Clark kinetic energy term for the given state object.

        Arguments:
            zf (State): Full state object (not filtered)

        Returns:
            cek (FieldVariable): Field containing the Clark kinetic energy term
        """
        # transform to physical space if not already
        if zf.is_spectral:
            zf = zf.fft()

        cek = self.e_kin_corr(self.clark_correction(zf),
                              self.filter_state(zf))
        return cek

    def clark_epot(self, zf:State):
        """
        Computes the Clark potential energy term for the given state object.

        Arguments:
            zf (State): Full state object (not filtered)

        Returns:
            cep (FieldVariable): Field containing the Clark potential energy term
        """
        # transform to physical space if not already
        if zf.is_spectral:
            zf = zf.fft()
        
        cep = self.e_pot_corr(self.clark_correction(zf),
                              self.filter_state(zf))

        return cep
    
    def clark_etot(self, zf:State):
        """
        Computes the Clark total energy term for the given state object.

        Arguments:
            zf (State): Full state object (not filtered)

        Returns:
            cet (FieldVariable): Field containing the Clark total energy term
        """
        # etot contribution
        cet = self.clark_ekin(zf) + self.clark_epot(zf)
        cet.name = "Transfer to unresolved Etot"

        return cet
    
    def reynolds_ekin(self, zf:State):
        """
        Computes the Reynolds kinetic energy term for the given state object.

        Arguments:
            zf (State): Full state object (not filtered)

        Returns:
            rek (FieldVariable): Field containing the Reynolds kinetic energy term
        """
        # transform to physical space if not already
        if zf.is_spectral:
            zf = zf.fft()

        rek = self.e_kin_corr(self.reynolds_correction(zf),
                              self.filter_state(zf))
        return rek
    
    def reynolds_epot(self, zf:State):
        """
        Computes the Reynolds potential energy term for the given state object.

        Arguments:
            zf (State): Full state object (not filtered)

        Returns:
            rep (FieldVariable): Field containing the Reynolds potential energy term
        """
        # transform to physical space if not already
        if zf.is_spectral:
            zf = zf.fft()

        rep = self.e_pot_corr(self.reynolds_correction(zf),
                              self.filter_state(zf))
        return rep
    
    def reynolds_etot(self, zf:State):
        """
        Computes the Reynolds total energy term for the given state object.

        Arguments:
            zf (State): Full state object (not filtered)

        Returns:
            ret (FieldVariable): Field containing the Reynolds total energy term
        """
        # etot contribution
        ret = self.reynolds_ekin(zf) + self.reynolds_epot(zf)
        ret.name = "Transfer to unresolved Etot"

        return ret

    def diagnose_energy(self, zf):
        """
        Prints energy diagnostics for the given state object.
        """
        cp = zf.cp
        z = self.filter_state(zf)
        zr = zf - z
        integral = lambda f: cp.sum(f) * self.mset.dx * self.mset.dy
        print("======================== ENERGY DIAGNOSTICS ========================")
        print("            Resolved     |      Unresolved     |    Total")
        epotf = integral(zf.epot())
        epot = integral(z.epot())
        epotr = epotf - epot
        print("Pot: {:9.2e} ({:7.2%}) | {:9.2e} ({:7.2%}) | {:10.3e}".format(
            epot, epot/epotf, epotr, epotr/epotf, epotf))
        ekinf = integral(zf.ekin())
        ekin = integral(z.ekin())
        ekinr = ekinf - ekin
        print("Kin: {:9.2e} ({:7.2%}) | {:9.2e} ({:7.2%}) | {:10.3e}".format(
            ekin, ekin/ekinf, ekinr, ekinr/ekinf, ekinf))
        etotf = integral(zf.etot())
        etot = integral(z.etot())
        etotr = etotf - etot
        print("Tot: {:9.2e} ({:7.2%}) | {:9.2e} ({:7.2%}) | {:10.3e}".format(
            etot, etot/etotf, etotr, etotr/etotf, etotf))
        print("------------------------- ENERGY TRANSFER --------------------------")
        print("            Leonard      |         Clark       |         Reynolds")
        epotl = integral(self.leonard_epot(zf))
        epotc = integral(self.clark_epot(zf))
        epotr = integral(self.reynolds_epot(zf))
        epotf = epotl + epotc + epotr
        print("Pot: {:9.2e} ({:7.2%}) | {:9.2e} ({:7.2%}) | {:10.3e} ({:7.2%})".format(
            epotl, epotl/epotf, epotc, epotc/epotf, epotr, epotr/epotf))
        ekinl = integral(self.leonard_ekin(zf))
        ekinc = integral(self.clark_ekin(zf))
        ekinr = integral(self.reynolds_ekin(zf))
        ekinf = ekinl + ekinc + ekinr
        print("Kin: {:9.2e} ({:7.2%}) | {:9.2e} ({:7.2%}) | {:10.3e} ({:7.2%})".format(
            ekinl, ekinl/ekinf, ekinc, ekinc/ekinf, ekinr, ekinr/ekinf))
        etotl = integral(self.leonard_etot(zf))
        etotc = integral(self.clark_etot(zf))
        etotr = integral(self.reynolds_etot(zf))
        etotf = etotl + etotc + etotr
        print("Tot: {:9.2e} ({:7.2%}) | {:9.2e} ({:7.2%}) | {:10.3e} ({:7.2%})".format(
            etotl, etotl/etotf, etotc, etotc/etotf, etotr, etotr/etotf))
        print("====================================================================")
        return

# remove symbols from namespace
del ModelSettings, Grid, State, FieldVariable