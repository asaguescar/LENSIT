import sncosmo


def get_saltmodel(mwebv=None):
    """ SALT2 model incl dust correction """
    dust = sncosmo.F99Dust()
    model = sncosmo.Model("salt2", effects=[dust],
                          effect_names=['mw'],
                          effect_frames=['obs'])
    if mwebv is not None:
        model.set(mwebv=mwebv)

    return model


def fit_lc(data, mwebv=0.1558, redshift=0.35440, modelcov=False, **kwargs):
    """ fit a lightcurve given a pandas Dataframe """

    # get the salt2 + MW dust model.
    model = get_saltmodel(mwebv=mwebv)
    model.set(z=redshift)

    # data as sncosmo input.
    parameters = ['t0', 'x0', 'x1', 'c']

    # fit salt
    # print(model)
    result, fitted_model = sncosmo.fit_lc(data, model, parameters, modelcov=modelcov, **kwargs)

    return result, fitted_model


def lc_salt2_fit(lc, mwebv, zlens, doPlot=True, filename=''):
    from astropy.table import Table
    ''' 
    Fit salt2 model to a lightcurve
    '''
    out = {}
    data = Table.from_pandas(lc[['time', 'zp', 'zpsys',
                                 'band', 'flux', 'fluxerr']])
    try:
        result, fitted_model = fit_lc(data, mwebv=mwebv, redshift=zlens)
        if doPlot:
            import matplotlib.pyplot as plt
            _ = sncosmo.plot_lc(data, model=fitted_model, errors=result.errors)
            plt.savefig('plots/' + filename)
            plt.close()

        if result.ndof != 0:
            out['mb_fit'] = float(fitted_model.source_peakabsmag('bessellb', 'ab'))
            out['t0lens'] = float(fitted_model.parameters[1])
            out['x0lens'] = float(fitted_model.parameters[2])
            out['x1lens'] = float(fitted_model.parameters[3])
            out['clens'] = float(fitted_model.parameters[4])
            out['errt0lens'] = result.errors['t0']
            out['errx0lens'] = result.errors['x0']
            out['errx1lens'] = result.errors['x1']
            out['errclens'] = result.errors['c']
            out['chisqlens'] = result.chisq / result.ndof
            out['ndof'] = result.ndof
        elif result.ndof == 0:
            out['mb_fit'] = float(fitted_model.source_peakabsmag('bessellb', 'ab'))
            out['t0lens'] = float(fitted_model.parameters[1])
            out['x0lens'] = float(fitted_model.parameters[2])
            out['x1lens'] = float(fitted_model.parameters[3])
            out['clens'] = float(fitted_model.parameters[4])
            out['errt0lens'] = result.errors['t0']
            out['errx0lens'] = result.errors['x0']
            out['errx1lens'] = result.errors['x1']
            out['errclens'] = result.errors['c']
            out['chisqlens'] = 999
            out['ndof'] = 999
    except:
        out['mb_fit'] = 999
        out['t0lens'] = 999
        out['x0lens'] = 999
        out['x1lens'] = 999
        out['clens'] = 999
        out['errt0lens'] = 999
        out['errx0lens'] = 999
        out['errx1lens'] = 999
        out['errclens'] = 999
        out['chisqlens'] = 999
        out['ndof'] = 999

    return out