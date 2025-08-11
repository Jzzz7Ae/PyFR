<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.entropy'/>

<%pyfr:macro name='get_minima' params='u, m0, dmin, pmin, emin'>
    fpdtype_t d, p, e;
    fpdtype_t ui[${nvars}];

    dmin = ${fpdtype_max}; pmin = ${fpdtype_max}; emin = ${fpdtype_max};

    // Interior points
    for (int i = 0; i < ${nupts}; i++)
    {
    % for j in range(nvars):
        ui[${j}] = u[i][${j}];
    % endfor
        ${pyfr.expand('compute_entropy', 'ui', 'd', 'p', 'e')};
        dmin = fmin(dmin, d); pmin = fmin(pmin, p); emin = fmin(emin, e);
    }

    // Face points if needed
    % if not fpts_in_upts:
    fpdtype_t uf[${nvars}];
    for (int fidx = 0; fidx < ${nfpts}; fidx++)
    {
        % for vidx in range(nvars):
        uf[${vidx}] = ${pyfr.dot('m0[fidx][{k}]', f'u[{{k}}][{vidx}]', k=nupts)};
        % endfor
        ${pyfr.expand('compute_entropy', 'uf', 'd', 'p', 'e')};
        dmin = fmin(dmin, d); pmin = fmin(pmin, p); emin = fmin(emin, e);
    }
    % endif
</%pyfr:macro>

<%pyfr:macro name='apply_filter_single' params='up, f, d, p, e'>
    fpdtype_t ui[${nvars}];
% for vidx in range(nvars):
    ui[${vidx}] = up[0][${vidx}];
% endfor

    fpdtype_t v = 1.0, v2 = 1.0;
    for (int pidx = 1; pidx < ${order+1}; pidx++)
    {
        v2 *= v*v*f;
        v  *= f;
        % for vidx in range(nvars):
        ui[${vidx}] += v2*up[pidx][${vidx}];
        % endfor
    }

    ${pyfr.expand('compute_entropy', 'ui', 'd', 'p', 'e')};
</%pyfr:macro>

<%pyfr:kernel name='entropy_sensor_artificial_viscosity' ndim='1'
              u='in fpdtype_t[${str(nupts)}][${str(nvars)}]'
              entmin_int='inout fpdtype_t[${str(nfaces)}]'
              vdm='in broadcast fpdtype_t[${str(nefpts)}][${str(nupts)}]'
              invvdm='in broadcast fpdtype_t[${str(nupts)}][${str(nupts)}]'
              m0='in broadcast fpdtype_t[${str(nfpts)}][${str(nupts)}]'
              artvisc='out fpdtype_t'
              zeta='inout fpdtype_t'>
    fpdtype_t dmin, pmin, emin;
    ${pyfr.expand('get_minima', 'u', 'm0', 'dmin', 'pmin', 'emin')};

    fpdtype_t entmin = ${fpdtype_max};
    for (int fidx = 0; fidx < ${nfaces}; ++fidx)
        entmin = fmin(entmin, entmin_int[fidx]);

    fpdtype_t sev = 0.0;

    if (dmin < ${d_min} || pmin < ${p_min} || emin < entmin - ${e_tol})
    {
        fpdtype_t umodes[${nupts}][${nvars}];
        for (int uidx = 0; uidx < ${nupts}; ++uidx)
            for (int vidx = 0; vidx < ${nvars}; ++vidx)
                umodes[uidx][vidx] = ${pyfr.dot('invvdm[uidx][{k}]', 'u[{k}][vidx]', k=nupts)};

        fpdtype_t f = 1.0, f_low, f_high, fnew;
        fpdtype_t d, p, e;
        fpdtype_t up[${order+1}][${nvars}];

        for (int uidx = 0; uidx < ${nefpts}; ++uidx)
        {
            % for pidx, vidx in pyfr.ndrange(order+1, nvars):
            up[${pidx}][${vidx}] = (${' + '.join(f'vdm[uidx][{k}]*umodes[{k}][{vidx}]'
                                                   for k, dd in enumerate(ubdegs) if dd == pidx)});
            % endfor

            ${pyfr.expand('apply_filter_single', 'up', 'f', 'd', 'p', 'e')};

            if (d < ${d_min} || p < ${p_min} || e < entmin - ${e_tol})
            {
                f_high = f; f_low = 0.0;
                for (int iter = 0; iter < ${niters} && f_high - f_low > ${f_tol}; ++iter)
                {
                    fnew = 0.5*(f_low + f_high);
                    ${pyfr.expand('apply_filter_single', 'up', 'fnew', 'd', 'p', 'e')};
                    if (d < ${d_min} || p < ${p_min} || e < entmin - ${e_tol})
                        f_high = fnew;
                    else
                        f_low  = fnew;
                }
                f = f_low;
            }
            zeta = -log(fmax(f,1.0e-12));
        }

        sev = 1.0 - f;
    }else{
        zeta=0;
    }

    artvisc = ${c['max-artvisc']} * sev;

% for fidx in range(nfaces):
    entmin_int[${fidx}] = emin;
% endfor
</%pyfr:kernel>

