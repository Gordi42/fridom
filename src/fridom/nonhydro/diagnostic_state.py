from fridom.framework import utils
from fridom.framework.state_base import StateBase
from fridom.framework.field_variable import FieldVariable
from fridom.framework.grid import cartesian
from fridom.nonhydro.model_settings import ModelSettings

class DiagnosticState(StateBase):
    def __init__(self, mset: ModelSettings, is_spectral=False, field_list=None) -> None:
        from fridom.framework.field_variable import FieldVariable
        if field_list is None:
            p = FieldVariable(
                mset, 
                name="p", 
                long_name="Pressure",
                units="m²/s",
                is_spectral=is_spectral, 
                position=mset.grid.cell_center)

            div = FieldVariable(
                mset,
                name="div", 
                long_name="Divergence",
                units="1/s",
                is_spectral=is_spectral, 
                position=mset.grid.cell_center)

            field_list = [p, div]
        super().__init__(mset, field_list, is_spectral)
        self.constructor = DiagnosticState
        return

    @property
    def p(self) -> FieldVariable:
        return self.fields["p"]

    @p.setter
    def p(self, value: FieldVariable) -> None:
        self.fields["p"] = value
    
    @property
    def div(self) -> FieldVariable:
        return self.fields["div"]
    
    @div.setter
    def div(self, value: FieldVariable) -> None:
        self.fields["div"] = value


utils.jaxify_class(DiagnosticState)
