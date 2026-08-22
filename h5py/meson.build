py.extension_module(
  'api_types_ext',
  custom_target(
    command: [template_base_cmd, '@CURRENT_SOURCE_DIR@/api_types_ext.pxd.template'],
    output: 'api_types_ext.pxd',
    depends: config_tgt,
  ),
  install: true,
  dependencies: npy_dep,
  c_args: npy_nodeprapi,
  limited_api: py.language_version(),
)

helpers = '../build_helpers'

api_tgt = custom_target(
  'api',
  command: [
    find_program(helpers / 'api_gen.py'),
    '@CURRENT_SOURCE_DIR@/api_functions.txt',
    '--config',
    config_tgt[1].full_path(),
    '-o',
    '@OUTDIR@',
  ],
  output: [
    '_hdf5.pxd',
    'defs.pxd',
    'defs.pyx',
  ],
  # this part shouldn't be needed (custom_target is *supposed* to
  # automatically include dependencies from command:
  # This is likely a bug in meson that it doesn't work with an indexed target ?
  depends: config_tgt,
)
api_dep = declare_dependency(sources: api_tgt)

cython_args = ['-I' + meson.current_source_dir(), '-I' + meson.current_build_dir()]

py.extension_module(
  'api_types_hdf5',
  custom_target(
    command: [template_base_cmd, '@CURRENT_SOURCE_DIR@/api_types_hdf5.pxd.template'],
    output: 'api_types_hdf5.pxd',
    depends: config_tgt,
  ),
  subdir: 'h5py',
  install: true,
  dependencies: npy_dep,
  c_args: npy_nodeprapi,
  cython_args: cython_args,
  limited_api: py.language_version(),
)

py.extension_module(
  'h5r',
  [
    custom_target(
      command: [template_base_cmd, '@CURRENT_SOURCE_DIR@/h5r.pxd.template'],
      output: 'h5r.pxd',
      depends: config_tgt,
    ),
    'h5r.pyx',
  ],
  install: true,
  dependencies: [npy_dep, api_dep],
  c_args: npy_nodeprapi,
  cython_args: cython_args,
  limited_api: py.language_version(),
)

py.install_sources(
  '__init__.py',
)

# subdir('_hl')

hl = '_hl'
py.install_sources(
  hl / '__init__.py',
  hl / 'attrs.py',
  hl / 'base.py',
  hl / 'compat.py',
  hl / 'dataset.py',
  hl / 'dims.py',
  hl / 'files.py',
  hl / 'filters.py',
  hl / 'group.py',
  hl / 'selections.py',
  hl / 'selections2.py',
  hl / 'vds.py',
  subdir: hl,
)
