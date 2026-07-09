from models.regression import calc_hlfpa_effort, calc_ifpa_effort, calc_mfp_effort

def test_regression_models():
    hlfpa_eff = calc_hlfpa_effort(10, 5, 5, 5, 5)
    assert round(hlfpa_eff, 2) == 1421.91

    ifpa_eff = calc_ifpa_effort(10, 5)
    assert round(ifpa_eff, 2) == 2800.96

    mfp_eff = calc_mfp_effort(10, 5, 5, 5, 5)
    assert round(mfp_eff, 2) == 587.22
