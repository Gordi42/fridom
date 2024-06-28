from fridom.framework.state_base import StateBase
from fridom.framework.field_variable import FieldVariable
from fridom.nonhydro.model_settings import ModelSettings

class DiagnosticState(StateBase):
    def __init__(self, mset: ModelSettings, is_spectral=False, field_list=None) -> None:
        from fridom.framework.field_variable import FieldVariable
        if field_list is None:
            p = FieldVariable(mset, 
                name="Pressure p", is_spectral=is_spectral)
            div = FieldVariable(mset,
                name="Divergence", is_spectral=is_spectral)
            field_list = [p, div]
        super().__init__(mset, field_list, is_spectral)
        self.constructor = DiagnosticState
        return

    @property
    def p(self) -> FieldVariable:
        return self.field_list[0]

    @p.setter
    def p(self, value: FieldVariable) -> None:
        self.field_list[0] = value
    
    @property
    def div(self) -> FieldVariable:
        return self.field_list[1]
    
    @div.setter
    def div(self, value: FieldVariable) -> None:
        self.field_list[1] = value