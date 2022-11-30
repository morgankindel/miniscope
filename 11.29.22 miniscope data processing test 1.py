#!/usr/bin/env python
# coding: utf-8

# In[25]:


get_ipython().run_cell_magic('capture', '', '%load_ext autoreload\n%autoreload 2\nimport itertools as itt\nimport os\nimport sys\n\nimport holoviews as hv\nimport numpy as np\nimport xarray as xr\nfrom dask.distributed import Client, LocalCluster\nfrom holoviews.operation.datashader import datashade, regrid\nfrom holoviews.util import Dynamic\nfrom IPython.core.display import display')


# In[26]:


# Set up Initial Basic Parameters#
minian_path = "."
dpath = "./demo_movies/"
minian_ds_path = os.path.join(dpath, "minian")
intpath = "./minian_intermediate"
subset = dict(frame=slice(0, None))
subset_mc = None
interactive = True
output_size = 100
n_workers = int(os.getenv("MINIAN_NWORKERS", 4))
param_save_minian = {
    "dpath": minian_ds_path,
    "meta_dict": dict(session=-1, animal=-2),
    "overwrite": True,
}

# Pre-processing Parameters#
param_load_videos = {
    "pattern": "msCam[0-9]+\.avi$",
    "dtype": np.uint8,
    "downsample": dict(frame=1, height=1, width=1),
    "downsample_strategy": "subset",
}
param_denoise = {"method": "median", "ksize": 7}
param_background_removal = {"method": "tophat", "wnd": 15}

# Motion Correction Parameters#
subset_mc = None
param_estimate_motion = {"dim": "frame"}

# Initialization Parameters#
param_seeds_init = {
    "wnd_size": 1000,
    "method": "rolling",
    "stp_size": 500,
    "max_wnd": 15,
    "diff_thres": 3,
}
param_pnr_refine = {"noise_freq": 0.06, "thres": 1}
param_ks_refine = {"sig": 0.05}
param_seeds_merge = {"thres_dist": 10, "thres_corr": 0.8, "noise_freq": 0.06}
param_initialize = {"thres_corr": 0.8, "wnd": 10, "noise_freq": 0.06}
param_init_merge = {"thres_corr": 0.8}

# CNMF Parameters#
param_get_noise = {"noise_range": (0.06, 0.5)}
param_first_spatial = {
    "dl_wnd": 10,
    "sparse_penal": 0.01,
    "size_thres": (25, None),
}
param_first_temporal = {
    "noise_freq": 0.06,
    "sparse_penal": 1,
    "p": 1,
    "add_lag": 20,
    "jac_thres": 0.2,
}
param_first_merge = {"thres_corr": 0.8}
param_second_spatial = {
    "dl_wnd": 10,
    "sparse_penal": 0.01,
    "size_thres": (25, None),
}
param_second_temporal = {
    "noise_freq": 0.06,
    "sparse_penal": 1,
    "p": 1,
    "add_lag": 20,
    "jac_thres": 0.4,
}

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MINIAN_INTERMEDIATE"] = intpath


# In[27]:


get_ipython().run_cell_magic('capture', '', 'sys.path.append(minian_path)\nfrom minian.cnmf import (\n    compute_AtC,\n    compute_trace,\n    get_noise_fft,\n    smooth_sig,\n    unit_merge,\n    update_spatial,\n    update_temporal,\n    update_background,\n)\nfrom minian.initialization import (\n    gmm_refine,\n    initA,\n    initC,\n    intensity_refine,\n    ks_refine,\n    pnr_refine,\n    seeds_init,\n    seeds_merge,\n)\nfrom minian.motion_correction import apply_transform, estimate_motion\nfrom minian.preprocessing import denoise, remove_background\nfrom minian.utilities import (\n    TaskAnnotation,\n    get_optimal_chk,\n    load_videos,\n    open_minian,\n    save_minian,\n)\nfrom minian.visualization import (\n    CNMFViewer,\n    VArrayViewer,\n    generate_videos,\n    visualize_gmm_fit,\n    visualize_motion,\n    visualize_preprocess,\n    visualize_seeds,\n    visualize_spatial_update,\n    visualize_temporal_update,\n    write_video,\n)')


# In[28]:


dpath = os.path.abspath(dpath)
hv.notebook_extension("bokeh", width=100)


# In[29]:


cluster = LocalCluster(
    n_workers=n_workers,
    memory_limit="2GB",
    resources={"MEM": 1},
    threads_per_worker=2,
    dashboard_address=":8787",
)
annt_plugin = TaskAnnotation()
cluster.scheduler.add_plugin(annt_plugin)
client = Client(cluster)


# In[30]:


#param_load_videos["vapth"] = "C:/Users/lspadmin/Documents/miniscopeDeviceName"


# In[36]:


param_load_videos


# In[37]:


varr = load_videos(dpath, **param_load_videos)
chk, _ = get_optimal_chk(varr, dtype=float)


# In[38]:


get_ipython().run_cell_magic('time', '', 'varr = save_minian(\n    varr.chunk({"frame": chk["frame"], "height": -1, "width": -1}).rename("varr"),\n    intpath,\n    overwrite=True,\n)')


# In[34]:


varr


# In[152]:


hv.output(size=output_size)
if interactive:
    vaviewer = VArrayViewer(varr, framerate=5, summary=["mean", "max", "diff"])
    display(vaviewer.show())


# In[153]:


if interactive:
    try:
        subset_mc = list(vaviewer.mask.values())[0]
    except IndexError:
        pass


# In[154]:


get_ipython().run_cell_magic('time', '', 'subset = None\nvarr_ref = varr.sel(subset)\nvarr_min = varr_ref.min("frame").compute()\nvarr_ref = varr_ref - varr_min')


# In[155]:


subset_mc


# In[156]:


hv.output(size=int(output_size * 0.7))
if interactive:
    vaviewer = VArrayViewer(
        [varr.rename("original"), varr_ref.rename("glow_removed")],
        framerate=5,
        summary=None,
        layout=True,
    )
    display(vaviewer.show())


# In[157]:


param_denoise


# In[112]:


hv.output(size=int(output_size * 0.6))
if interactive:
    display(
        visualize_preprocess(
            varr_ref.isel(frame=0).compute(),
            denoise,
            method=["median"],
            ksize=[5, 7, 9],
        )
    )


# In[158]:


varr_ref = denoise(varr_ref, **param_denoise)


# In[159]:


param_background_removal


# In[160]:


hv.output(size=int(output_size * 0.6))
if interactive:
    display(
        visualize_preprocess(
            varr_ref.isel(frame=0).compute(),
            remove_background,
            method=["tophat"],
            wnd=[10, 15, 20],
        )
    )


# In[161]:


varr_ref = remove_background(varr_ref, **param_background_removal)


# In[162]:


get_ipython().run_cell_magic('time', '', 'varr_ref = save_minian(varr_ref.rename("varr_ref"), dpath=intpath, overwrite=True)')


# In[163]:


param_estimate_motion


# In[164]:


get_ipython().run_cell_magic('time', '', 'motion = estimate_motion(varr_ref.sel(subset_mc), **param_estimate_motion)')


# In[148]:


motion


# In[165]:


param_save_minian


# In[166]:


get_ipython().run_cell_magic('time', '', 'motion = save_minian(\n    motion.rename("motion").chunk({"frame": chk["frame"]}), **param_save_minian\n)')


# In[147]:


motion


# In[167]:


hv.output(size=output_size)
visualize_motion(motion)


# In[170]:


Y = apply_transform(varr_ref, motion, fill=0)


# In[171]:


get_ipython().run_cell_magic('time', '', 'Y_fm_chk = save_minian(Y.astype(float).rename("Y_fm_chk"), intpath, overwrite=True)\nY_hw_chk = save_minian(\n    Y_fm_chk.rename("Y_hw_chk"),\n    intpath,\n    overwrite=True,\n    chunks={"frame": -1, "height": chk["height"], "width": chk["width"]},\n)')


# In[172]:


hv.output(size=int(output_size * 0.7))
if interactive:
    vaviewer = VArrayViewer(
        [varr_ref.rename("before_mc"), Y_fm_chk.rename("after_mc")],
        framerate=5,
        summary=None,
        layout=True,
    )
    display(vaviewer.show())


# In[173]:


im_opts = dict(
    frame_width=500,
    aspect=varr_ref.sizes["width"] / varr_ref.sizes["height"],
    cmap="Viridis",
    colorbar=True,
)
(
    regrid(
        hv.Image(
            varr_ref.max("frame").compute().astype(np.float32),
            ["width", "height"],
            label="before_mc",
        ).opts(**im_opts)
    )
    + regrid(
        hv.Image(
            Y_hw_chk.max("frame").compute().astype(np.float32),
            ["width", "height"],
            label="after_mc",
        ).opts(**im_opts)
    )
)


# In[174]:


get_ipython().run_cell_magic('time', '', 'vid_arr = xr.concat([varr_ref, Y_fm_chk], "width").chunk({"width": -1})\nwrite_video(vid_arr, "minian_mc.mp4", dpath)')


# In[175]:


max_proj = save_minian(
    Y_fm_chk.max("frame").rename("max_proj"), **param_save_minian
).compute()


# In[176]:


param_seeds_init


# In[177]:


get_ipython().run_cell_magic('time', '', 'seeds = seeds_init(Y_fm_chk, **param_seeds_init)')


# In[178]:


seeds.head()


# In[179]:


hv.output(size=output_size)
visualize_seeds(max_proj, seeds)


# In[180]:


get_ipython().run_cell_magic('time', '', 'if interactive:\n    noise_freq_list = [0.005, 0.01, 0.02, 0.06, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8]\n    example_seeds = seeds.sample(6, axis="rows")\n    example_trace = Y_hw_chk.sel(\n        height=example_seeds["height"].to_xarray(),\n        width=example_seeds["width"].to_xarray(),\n    ).rename(**{"index": "seed"})\n    smooth_dict = dict()\n    for freq in noise_freq_list:\n        trace_smth_low = smooth_sig(example_trace, freq)\n        trace_smth_high = smooth_sig(example_trace, freq, btype="high")\n        trace_smth_low = trace_smth_low.compute()\n        trace_smth_high = trace_smth_high.compute()\n        hv_trace = hv.HoloMap(\n            {\n                "signal": (\n                    hv.Dataset(trace_smth_low)\n                    .to(hv.Curve, kdims=["frame"])\n                    .opts(frame_width=300, aspect=2, ylabel="Signal (A.U.)")\n                ),\n                "noise": (\n                    hv.Dataset(trace_smth_high)\n                    .to(hv.Curve, kdims=["frame"])\n                    .opts(frame_width=300, aspect=2, ylabel="Signal (A.U.)")\n                ),\n            },\n            kdims="trace",\n        ).collate()\n        smooth_dict[freq] = hv_trace')


# In[181]:


hv.output(size=int(output_size * 0.7))
if interactive:
    hv_res = (
        hv.HoloMap(smooth_dict, kdims=["noise_freq"])
        .collate()
        .opts(aspect=2)
        .overlay("trace")
        .layout("seed")
        .cols(3)
    )
    display(hv_res)


# In[182]:


param_pnr_refine


# In[183]:


get_ipython().run_cell_magic('time', '', 'seeds, pnr, gmm = pnr_refine(Y_hw_chk, seeds, **param_pnr_refine)')


# In[187]:


seeds.head()


# In[185]:


if gmm:
    display(visualize_gmm_fit(pnr, gmm, 100))
else:
    print("nothing to show")


# In[186]:


hv.output(size=output_size)
visualize_seeds(max_proj, seeds, "mask_pnr")


# In[188]:


param_ks_refine


# In[189]:


get_ipython().run_cell_magic('time', '', 'seeds = ks_refine(Y_hw_chk, seeds, **param_ks_refine)')


# In[190]:


hv.output(size=output_size)
visualize_seeds(max_proj, seeds, "mask_ks")


# In[191]:


param_seeds_merge


# In[192]:


get_ipython().run_cell_magic('time', '', 'seeds_final = seeds[seeds["mask_ks"] & seeds["mask_pnr"]].reset_index(drop=True)\nseeds_final = seeds_merge(Y_hw_chk, max_proj, seeds_final, **param_seeds_merge)')


# In[193]:


hv.output(size=output_size)
visualize_seeds(max_proj, seeds_final, "mask_mrg")


# In[194]:


param_initialize


# In[195]:


get_ipython().run_cell_magic('time', '', 'A_init = initA(Y_hw_chk, seeds_final[seeds_final["mask_mrg"]], **param_initialize)\nA_init = save_minian(A_init.rename("A_init"), intpath, overwrite=True)')


# In[196]:


get_ipython().run_cell_magic('time', '', 'C_init = initC(Y_fm_chk, A_init)\nC_init = save_minian(\n    C_init.rename("C_init"), intpath, overwrite=True, chunks={"unit_id": 1, "frame": -1}\n)')


# In[197]:


param_init_merge


# In[198]:


get_ipython().run_cell_magic('time', '', 'A, C = unit_merge(A_init, C_init, **param_init_merge)\nA = save_minian(A.rename("A"), intpath, overwrite=True)\nC = save_minian(C.rename("C"), intpath, overwrite=True)\nC_chk = save_minian(\n    C.rename("C_chk"),\n    intpath,\n    overwrite=True,\n    chunks={"unit_id": -1, "frame": chk["frame"]},\n)')


# In[199]:


get_ipython().run_cell_magic('time', '', 'b, f = update_background(Y_fm_chk, A, C_chk)\nf = save_minian(f.rename("f"), intpath, overwrite=True)\nb = save_minian(b.rename("b"), intpath, overwrite=True)')


# In[200]:


hv.output(size=int(output_size * 0.55))
im_opts = dict(
    frame_width=500,
    aspect=A.sizes["width"] / A.sizes["height"],
    cmap="Viridis",
    colorbar=True,
)
cr_opts = dict(frame_width=750, aspect=1.5 * A.sizes["width"] / A.sizes["height"])
(
    regrid(
        hv.Image(
            A.max("unit_id").rename("A").compute().astype(np.float32),
            kdims=["width", "height"],
        ).opts(**im_opts)
    ).relabel("Initial Spatial Footprints")
    + regrid(
        hv.Image(
            C.rename("C").compute().astype(np.float32), kdims=["frame", "unit_id"]
        ).opts(cmap="viridis", colorbar=True, **cr_opts)
    ).relabel("Initial Temporal Components")
    + regrid(
        hv.Image(
            b.rename("b").compute().astype(np.float32), kdims=["width", "height"]
        ).opts(**im_opts)
    ).relabel("Initial Background Sptial")
    + datashade(hv.Curve(f.rename("f").compute(), kdims=["frame"]), min_alpha=200)
    .opts(**cr_opts)
    .relabel("Initial Background Temporal")
).cols(2)


# In[201]:


param_get_noise


# In[202]:


get_ipython().run_cell_magic('time', '', 'sn_spatial = get_noise_fft(Y_hw_chk, **param_get_noise)\nsn_spatial = save_minian(sn_spatial.rename("sn_spatial"), intpath, overwrite=True)')


# In[204]:


if interactive:
    units = np.random.choice(A.coords["unit_id"], 10, replace=False)
    units.sort()
    A_sub = A.sel(unit_id=units).persist()
    C_sub = C.sel(unit_id=units).persist()


# In[207]:


get_ipython().run_cell_magic('time', '', 'if interactive:\n    sprs_ls = [0.005, 0.01, 0.03, 0.05]\n    A_dict = dict()\n    C_dict = dict()\n    for cur_sprs in sprs_ls:\n        cur_A, cur_mask, cur_norm = update_spatial(\n            Y_hw_chk,\n            A_sub,\n            C_sub,\n            sn_spatial,\n            in_memory=True,\n            dl_wnd=param_first_spatial["dl_wnd"],\n            sparse_penal=cur_sprs,\n        )\n        if cur_A.sizes["unit_id"]:\n            A_dict[cur_sprs] = cur_A.compute()\n            C_dict[cur_sprs] = C_sub.sel(unit_id=cur_mask).compute()\n    hv_res = visualize_spatial_update(A_dict, C_dict, kdims=["sparse penalty"])')


# In[208]:


hv.output(size=int(output_size * 0.6))
if interactive:
    display(hv_res)


# In[209]:


param_first_spatial


# In[210]:


get_ipython().run_cell_magic('time', '', 'A_new, mask, norm_fac = update_spatial(\n    Y_hw_chk, A, C, sn_spatial, **param_first_spatial\n)\nC_new = save_minian(\n    (C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True\n)\nC_chk_new = save_minian(\n    (C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"), intpath, overwrite=True\n)')


# In[211]:


get_ipython().run_cell_magic('time', '', 'b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)')


# In[212]:


hv.output(size=int(output_size * 0.6))
opts = dict(
    plot=dict(height=A.sizes["height"], width=A.sizes["width"], colorbar=True),
    style=dict(cmap="Viridis"),
)
(
    regrid(
        hv.Image(
            A.max("unit_id").compute().astype(np.float32).rename("A"),
            kdims=["width", "height"],
        ).opts(**opts)
    ).relabel("Spatial Footprints Initial")
    + regrid(
        hv.Image(
            (A.fillna(0) > 0).sum("unit_id").compute().astype(np.uint8).rename("A"),
            kdims=["width", "height"],
        ).opts(**opts)
    ).relabel("Binary Spatial Footprints Initial")
    + regrid(
        hv.Image(
            A_new.max("unit_id").compute().astype(np.float32).rename("A"),
            kdims=["width", "height"],
        ).opts(**opts)
    ).relabel("Spatial Footprints First Update")
    + regrid(
        hv.Image(
            (A_new > 0).sum("unit_id").compute().astype(np.uint8).rename("A"),
            kdims=["width", "height"],
        ).opts(**opts)
    ).relabel("Binary Spatial Footprints First Update")
).cols(2)


# In[213]:


hv.output(size=int(output_size * 0.55))
opts_im = dict(
    plot=dict(height=b.sizes["height"], width=b.sizes["width"], colorbar=True),
    style=dict(cmap="Viridis"),
)
opts_cr = dict(plot=dict(height=b.sizes["height"], width=b.sizes["height"] * 2))
(
    regrid(
        hv.Image(b.compute().astype(np.float32), kdims=["width", "height"]).opts(
            **opts_im
        )
    ).relabel("Background Spatial Initial")
    + hv.Curve(f.compute().rename("f").astype(np.float16), kdims=["frame"])
    .opts(**opts_cr)
    .relabel("Background Temporal Initial")
    + regrid(
        hv.Image(b_new.compute().astype(np.float32), kdims=["width", "height"]).opts(
            **opts_im
        )
    ).relabel("Background Spatial First Update")
    + hv.Curve(f_new.compute().rename("f").astype(np.float16), kdims=["frame"])
    .opts(**opts_cr)
    .relabel("Background Temporal First Update")
).cols(2)


# In[214]:


get_ipython().run_cell_magic('time', '', 'A = save_minian(\n    A_new.rename("A"),\n    intpath,\n    overwrite=True,\n    chunks={"unit_id": 1, "height": -1, "width": -1},\n)\nb = save_minian(b_new.rename("b"), intpath, overwrite=True)\nf = save_minian(\n    f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True\n)\nC = save_minian(C_new.rename("C"), intpath, overwrite=True)\nC_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)')


# In[215]:


if interactive:
    units = np.random.choice(A.coords["unit_id"], 10, replace=False)
    units.sort()
    A_sub = A.sel(unit_id=units).persist()
    C_sub = C_chk.sel(unit_id=units).persist()


# In[216]:


get_ipython().run_cell_magic('time', '', 'if interactive:\n    p_ls = [1]\n    sprs_ls = [0.1, 0.5, 1, 2]\n    add_ls = [20]\n    noise_ls = [0.06]\n    YA_dict, C_dict, S_dict, g_dict, sig_dict, A_dict = [dict() for _ in range(6)]\n    YrA = (\n        compute_trace(Y_fm_chk, A_sub, b, C_sub, f)\n        .persist()\n        .chunk({"unit_id": 1, "frame": -1})\n    )\n    for cur_p, cur_sprs, cur_add, cur_noise in itt.product(\n        p_ls, sprs_ls, add_ls, noise_ls\n    ):\n        ks = (cur_p, cur_sprs, cur_add, cur_noise)\n        print(\n            "p:{}, sparse penalty:{}, additional lag:{}, noise frequency:{}".format(\n                cur_p, cur_sprs, cur_add, cur_noise\n            )\n        )\n        cur_C, cur_S, cur_b0, cur_c0, cur_g, cur_mask = update_temporal(\n            A_sub,\n            C_sub,\n            YrA=YrA,\n            sparse_penal=cur_sprs,\n            p=cur_p,\n            use_smooth=True,\n            add_lag=cur_add,\n            noise_freq=cur_noise,\n        )\n        YA_dict[ks], C_dict[ks], S_dict[ks], g_dict[ks], sig_dict[ks], A_dict[ks] = (\n            YrA.compute(),\n            cur_C.compute(),\n            cur_S.compute(),\n            cur_g.compute(),\n            (cur_C + cur_b0 + cur_c0).compute(),\n            A_sub.compute(),\n        )\n    hv_res = visualize_temporal_update(\n        YA_dict,\n        C_dict,\n        S_dict,\n        g_dict,\n        sig_dict,\n        A_dict,\n        kdims=["p", "sparse penalty", "additional lag", "noise frequency"],\n    )')


# In[217]:


hv.output(size=int(output_size * 0.6))
if interactive:
    display(hv_res)


# In[218]:


get_ipython().run_cell_magic('time', '', 'YrA = save_minian(\n    compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"),\n    intpath,\n    overwrite=True,\n    chunks={"unit_id": 1, "frame": -1},\n)')


# In[219]:


param_first_temporal


# In[220]:


get_ipython().run_cell_magic('time', '', 'C_new, S_new, b0_new, c0_new, g, mask = update_temporal(\n    A, C, YrA=YrA, **param_first_temporal\n)')


# In[221]:


hv.output(size=int(output_size * 0.6))
opts_im = dict(frame_width=500, aspect=2, colorbar=True, cmap="Viridis")
(
    regrid(
        hv.Image(
            C.compute().astype(np.float32).rename("ci"), kdims=["frame", "unit_id"]
        ).opts(**opts_im)
    ).relabel("Temporal Trace Initial")
    + hv.Div("")
    + regrid(
        hv.Image(
            C_new.compute().astype(np.float32).rename("c1"), kdims=["frame", "unit_id"]
        ).opts(**opts_im)
    ).relabel("Temporal Trace First Update")
    + regrid(
        hv.Image(
            S_new.compute().astype(np.float32).rename("s1"), kdims=["frame", "unit_id"]
        ).opts(**opts_im)
    ).relabel("Spikes First Update")
).cols(2)


# In[222]:


hv.output(size=int(output_size * 0.6))
if interactive:
    h, w = A.sizes["height"], A.sizes["width"]
    im_opts = dict(aspect=w / h, frame_width=500, cmap="Viridis")
    cr_opts = dict(aspect=3, frame_width=1000)
    bad_units = mask.where(mask == False, drop=True).coords["unit_id"].values
    if len(bad_units) > 0:
        hv_res = (
            hv.NdLayout(
                {
                    "Spatial Footprint": Dynamic(
                        hv.Dataset(A.sel(unit_id=bad_units).compute().rename("A"))
                        .to(hv.Image, kdims=["width", "height"])
                        .opts(**im_opts)
                    ),
                    "Spatial Footprints of Accepted Units": Dynamic(
                        hv.Image(
                            A.sel(unit_id=mask).sum("unit_id").compute().rename("A"),
                            kdims=["width", "height"],
                        ).opts(**im_opts)
                    ),
                }
            )
            + datashade(
                hv.Dataset(YrA.sel(unit_id=bad_units).rename("raw")).to(
                    hv.Curve, kdims=["frame"]
                )
            )
            .opts(**cr_opts)
            .relabel("Temporal Trace")
        ).cols(1)
        display(hv_res)
    else:
        print("No rejected units to display")


# In[223]:


hv.output(size=int(output_size * 0.6))
if interactive:
    sig = C_new + b0_new + c0_new
    display(
        visualize_temporal_update(
            YrA.sel(unit_id=mask),
            C_new,
            S_new,
            g,
            sig,
            A.sel(unit_id=mask),
        )
    )


# In[226]:


get_ipython().run_cell_magic('time', '', 'C = save_minian(\n    C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True\n)\nC_chk = save_minian(\n    C.rename("C_chk"),\n    intpath,\n    overwrite=True,\n    chunks={"unit_id": -1, "frame": chk["frame"]},\n)\nS = save_minian(\n    S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True\n)\nb0 = save_minian(\n    b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True\n)\nc0 = save_minian(\n    c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True\n)\nA = A.sel(unit_id=C.coords["unit_id"].values)')


# In[227]:


param_first_merge


# In[228]:


get_ipython().run_cell_magic('time', '', 'A_mrg, C_mrg, [sig_mrg] = unit_merge(A, C, [C + b0 + c0], **param_first_merge)')


# In[229]:


hv.output(size=int(output_size * 0.6))
opts_im = dict(frame_width=500, aspect=2, colorbar=True, cmap="Viridis")
(
    regrid(
        hv.Image(
            C.compute().astype(np.float32).rename("c1"), kdims=["frame", "unit_id"]
        )
        .relabel("Temporal Signals Before Merge")
        .opts(**opts_im)
    )
    + regrid(
        hv.Image(
            C_mrg.compute().astype(np.float32).rename("c2"), kdims=["frame", "unit_id"]
        )
        .relabel("Temporal Signals After Merge")
        .opts(**opts_im)
    )
)


# In[230]:


get_ipython().run_cell_magic('time', '', 'A = save_minian(A_mrg.rename("A_mrg"), intpath, overwrite=True)\nC = save_minian(C_mrg.rename("C_mrg"), intpath, overwrite=True)\nC_chk = save_minian(\n    C.rename("C_mrg_chk"),\n    intpath,\n    overwrite=True,\n    chunks={"unit_id": -1, "frame": chk["frame"]},\n)\nsig = save_minian(sig_mrg.rename("sig_mrg"), intpath, overwrite=True)')


# In[231]:


if interactive:
    units = np.random.choice(A.coords["unit_id"], 10, replace=False)
    units.sort()
    A_sub = A.sel(unit_id=units).persist()
    C_sub = sig.sel(unit_id=units).persist()


# In[232]:


get_ipython().run_cell_magic('time', '', 'if interactive:\n    sprs_ls = [5e-3, 1e-2, 5e-2]\n    A_dict = dict()\n    C_dict = dict()\n    for cur_sprs in sprs_ls:\n        cur_A, cur_mask, cur_norm = update_spatial(\n            Y_hw_chk,\n            A_sub,\n            C_sub,\n            sn_spatial,\n            in_memory=True,\n            dl_wnd=param_second_spatial["dl_wnd"],\n            sparse_penal=cur_sprs,\n        )\n        if cur_A.sizes["unit_id"]:\n            A_dict[cur_sprs] = cur_A.compute()\n            C_dict[cur_sprs] = C_sub.sel(unit_id=cur_mask).compute()\n    hv_res = visualize_spatial_update(A_dict, C_dict, kdims=["sparse penalty"])')


# In[233]:


hv.output(size=int(output_size * 0.6))
if interactive:
    display(hv_res)


# In[234]:


get_ipython().run_cell_magic('time', '', 'A_new, mask, norm_fac = update_spatial(\n    Y_hw_chk, A, C, sn_spatial, **param_second_spatial\n)\nC_new = save_minian(\n    (C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True\n)\nC_chk_new = save_minian(\n    (C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"), intpath, overwrite=True\n)')


# In[235]:


get_ipython().run_cell_magic('time', '', 'b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)')


# In[236]:


hv.output(size=int(output_size * 0.6))
opts = dict(
    plot=dict(height=A.sizes["height"], width=A.sizes["width"], colorbar=True),
    style=dict(cmap="Viridis"),
)
(
    regrid(
        hv.Image(
            A.max("unit_id").compute().astype(np.float32).rename("A"),
            kdims=["width", "height"],
        ).opts(**opts)
    ).relabel("Spatial Footprints Last")
    + regrid(
        hv.Image(
            (A.fillna(0) > 0).sum("unit_id").compute().astype(np.uint8).rename("A"),
            kdims=["width", "height"],
        ).opts(**opts)
    ).relabel("Binary Spatial Footprints Last")
    + regrid(
        hv.Image(
            A_new.max("unit_id").compute().astype(np.float32).rename("A"),
            kdims=["width", "height"],
        ).opts(**opts)
    ).relabel("Spatial Footprints New")
    + regrid(
        hv.Image(
            (A_new > 0).sum("unit_id").compute().astype(np.uint8).rename("A"),
            kdims=["width", "height"],
        ).opts(**opts)
    ).relabel("Binary Spatial Footprints New")
).cols(2)


# In[237]:


hv.output(size=int(output_size * 0.55))
opts_im = dict(
    plot=dict(height=b.sizes["height"], width=b.sizes["width"], colorbar=True),
    style=dict(cmap="Viridis"),
)
opts_cr = dict(plot=dict(height=b.sizes["height"], width=b.sizes["height"] * 2))
(
    regrid(
        hv.Image(b.compute().astype(np.float32), kdims=["width", "height"]).opts(
            **opts_im
        )
    ).relabel("Background Spatial Last")
    + hv.Curve(f.compute().rename("f").astype(np.float16), kdims=["frame"])
    .opts(**opts_cr)
    .relabel("Background Temporal Last")
    + regrid(
        hv.Image(b_new.compute().astype(np.float32), kdims=["width", "height"]).opts(
            **opts_im
        )
    ).relabel("Background Spatial New")
    + hv.Curve(f_new.compute().rename("f").astype(np.float16), kdims=["frame"])
    .opts(**opts_cr)
    .relabel("Background Temporal New")
).cols(2)


# In[238]:


get_ipython().run_cell_magic('time', '', 'A = save_minian(\n    A_new.rename("A"),\n    intpath,\n    overwrite=True,\n    chunks={"unit_id": 1, "height": -1, "width": -1},\n)\nb = save_minian(b_new.rename("b"), intpath, overwrite=True)\nf = save_minian(\n    f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True\n)\nC = save_minian(C_new.rename("C"), intpath, overwrite=True)\nC_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)')


# In[241]:


if interactive:
    units = np.random.choice(A.coords["unit_id"], 10, replace=False)
    units.sort()
    A_sub = A.sel(unit_id=units).persist()
    C_sub = C_chk.sel(unit_id=units).persist()


# In[242]:


get_ipython().run_cell_magic('time', '', 'if interactive:\n    p_ls = [1]\n    sprs_ls = [0.1, 0.5, 1, 2]\n    add_ls = [20]\n    noise_ls = [0.06]\n    YA_dict, C_dict, S_dict, g_dict, sig_dict, A_dict = [dict() for _ in range(6)]\n    YrA = (\n        compute_trace(Y_fm_chk, A_sub, b, C_sub, f)\n        .persist()\n        .chunk({"unit_id": 1, "frame": -1})\n    )\n    for cur_p, cur_sprs, cur_add, cur_noise in itt.product(\n        p_ls, sprs_ls, add_ls, noise_ls\n    ):\n        ks = (cur_p, cur_sprs, cur_add, cur_noise)\n        print(\n            "p:{}, sparse penalty:{}, additional lag:{}, noise frequency:{}".format(\n                cur_p, cur_sprs, cur_add, cur_noise\n            )\n        )\n        cur_C, cur_S, cur_b0, cur_c0, cur_g, cur_mask = update_temporal(\n            A_sub,\n            C_sub,\n            YrA=YrA,\n            sparse_penal=cur_sprs,\n            p=cur_p,\n            use_smooth=True,\n            add_lag=cur_add,\n            noise_freq=cur_noise,\n        )\n        YA_dict[ks], C_dict[ks], S_dict[ks], g_dict[ks], sig_dict[ks], A_dict[ks] = (\n            YrA.compute(),\n            cur_C.compute(),\n            cur_S.compute(),\n            cur_g.compute(),\n            (cur_C + cur_b0 + cur_c0).compute(),\n            A_sub.compute(),\n        )\n    hv_res = visualize_temporal_update(\n        YA_dict,\n        C_dict,\n        S_dict,\n        g_dict,\n        sig_dict,\n        A_dict,\n        kdims=["p", "sparse penalty", "additional lag", "noise frequency"],\n    )')


# In[243]:


hv.output(size=int(output_size * 0.6))
if interactive:
    display(hv_res)


# In[244]:


get_ipython().run_cell_magic('time', '', 'YrA = save_minian(\n    compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"),\n    intpath,\n    overwrite=True,\n    chunks={"unit_id": 1, "frame": -1},\n)')


# In[245]:


get_ipython().run_cell_magic('time', '', 'C_new, S_new, b0_new, c0_new, g, mask = update_temporal(\n    A, C, YrA=YrA, **param_second_temporal\n)')


# In[246]:


hv.output(size=int(output_size * 0.6))
opts_im = dict(frame_width=500, aspect=2, colorbar=True, cmap="Viridis")
(
    regrid(
        hv.Image(
            C.compute().astype(np.float32).rename("c1"), kdims=["frame", "unit_id"]
        ).opts(**opts_im)
    ).relabel("Temporal Trace Last")
    + regrid(
        hv.Image(
            S.compute().astype(np.float32).rename("s1"), kdims=["frame", "unit_id"]
        ).opts(**opts_im)
    ).relabel("Spikes Last")
    + regrid(
        hv.Image(
            C_new.compute().astype(np.float32).rename("c2"), kdims=["frame", "unit_id"]
        ).opts(**opts_im)
    ).relabel("Temporal Trace New")
    + regrid(
        hv.Image(
            S_new.compute().astype(np.float32).rename("s2"), kdims=["frame", "unit_id"]
        ).opts(**opts_im)
    ).relabel("Spikes New")
).cols(2)


# In[247]:


hv.output(size=int(output_size * 0.6))
if interactive:
    h, w = A.sizes["height"], A.sizes["width"]
    im_opts = dict(aspect=w / h, frame_width=500, cmap="Viridis")
    cr_opts = dict(aspect=3, frame_width=1000)
    bad_units = mask.where(mask == False, drop=True).coords["unit_id"].values
    if len(bad_units) > 0:
        hv_res = (
            hv.NdLayout(
                {
                    "Spatial Footprint": Dynamic(
                        hv.Dataset(A.sel(unit_id=bad_units).compute().rename("A"))
                        .to(hv.Image, kdims=["width", "height"])
                        .opts(**im_opts)
                    ),
                    "Spatial Footprints of Accepted Units": Dynamic(
                        hv.Image(
                            A.sel(unit_id=mask).sum("unit_id").compute().rename("A"),
                            kdims=["width", "height"],
                        ).opts(**im_opts)
                    ),
                }
            )
            + datashade(
                hv.Dataset(YrA.sel(unit_id=bad_units).rename("raw")).to(
                    hv.Curve, kdims=["frame"]
                )
            )
            .opts(**cr_opts)
            .relabel("Temporal Trace")
        ).cols(1)
        display(hv_res)
    else:
        print("No rejected units to display")


# In[248]:


hv.output(size=int(output_size * 0.6))
if interactive:
    sig = C_new + b0_new + c0_new
    display(
        visualize_temporal_update(
            YrA.sel(unit_id=mask),
            C_new,
            S_new,
            g,
            sig,
            A.sel(unit_id=mask),
        )
    )


# In[249]:


get_ipython().run_cell_magic('time', '', 'C = save_minian(\n    C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True\n)\nC_chk = save_minian(\n    C.rename("C_chk"),\n    intpath,\n    overwrite=True,\n    chunks={"unit_id": -1, "frame": chk["frame"]},\n)\nS = save_minian(\n    S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True\n)\nb0 = save_minian(\n    b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True\n)\nc0 = save_minian(\n    c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True\n)\nA = A.sel(unit_id=C.coords["unit_id"].values)')


# In[250]:


get_ipython().run_cell_magic('time', '', 'generate_videos(varr.sel(subset), Y_fm_chk, A=A, C=C_chk, vpath=dpath)')


# In[251]:


get_ipython().run_cell_magic('time', '', 'if interactive:\n    cnmfviewer = CNMFViewer(A=A, C=C, S=S, org=Y_fm_chk)')


# In[252]:


hv.output(size=int(output_size * 0.35))
if interactive:
    display(cnmfviewer.show())


# In[253]:


if interactive:
    A = A.assign_coords(unit_labels=("unit_id", cnmfviewer.unit_labels))
    C = C.assign_coords(unit_labels=("unit_id", cnmfviewer.unit_labels))
    S = S.assign_coords(unit_labels=("unit_id", cnmfviewer.unit_labels))
    c0 = c0.assign_coords(unit_labels=("unit_id", cnmfviewer.unit_labels))
    b0 = b0.assign_coords(unit_labels=("unit_id", cnmfviewer.unit_labels))


# In[254]:


get_ipython().run_cell_magic('time', '', 'A = save_minian(A.rename("A"), **param_save_minian)\nC = save_minian(C.rename("C"), **param_save_minian)\nS = save_minian(S.rename("S"), **param_save_minian)\nc0 = save_minian(c0.rename("c0"), **param_save_minian)\nb0 = save_minian(b0.rename("b0"), **param_save_minian)\nb = save_minian(b.rename("b"), **param_save_minian)\nf = save_minian(f.rename("f"), **param_save_minian)')


# In[255]:


client.close()
cluster.close()


# In[ ]:




