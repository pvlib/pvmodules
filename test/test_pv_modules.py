import pandas as pd
from pvmodules.cec_pv_modules import sam_cec_columns, convert_row
from numpy import nan
import pytest


@pytest.fixture(scope='session')
def cec_modules():
    modules = pd.read_csv('test/cec_pv_module_subset.csv')
    modules['Description'] = modules['Description'].replace(nan, '')
    return modules


def test_cec_to_sam(cec_modules):
    res = cec_modules.apply(convert_row, axis=1)
    assert set(res.columns).issubset(set(sam_cec_columns))
