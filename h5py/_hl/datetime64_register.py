from numpy import dtype
from ..h5t import register_dtype

dt_units = [
    # Basic datetime
    '',
    # Dates
    '[Y]', '[M]', '[D]',
    # Times
    '[h]', '[m]', '[s]', '[ms]', '[us]',
    '[ns]', '[ps]', '[fs]', '[as]',
]

for dt_kind in ['M8', 'm8']:
    for dt_unit in dt_units:
        for dt_order in ['<', '>']:
            register_dtype(dtype(dt_order + dt_kind + dt_unit))
