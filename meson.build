project(
  'h5py',
  'c',
  'cython',
  # https://mesonbuild.com/meson-python/reference/meson-compatibility.html#cmdoption-arg-1.6.0
  meson_version: '>=1.6.0',
)

# detect missing dependencies early
py = import('python').find_installation(pure: false)

npy_nodeprapi = [
  # keep in sync with pyproject.toml
  '-DNPY_NO_DEPRECATED_API=NPY_1_21_API_VERSION',
]

npy_dep = declare_dependency(
  dependencies: dependency('numpy'),
  compile_args: npy_nodeprapi,
)

if get_option('mpi')
  mpi_dep = declare_dependency(
    include_directories: dependency('mpi4py'),
  )
endif

helpers = 'build_helpers'

config_tgt = custom_target(
  'config',
  command: [find_program(helpers / 'config.py'), '-o', '@OUTDIR@'],
  output: [
    'h5config.json',
    'template_substitutions.json'
  ],
)
template_gen = generator(
  find_program(helpers / 'template.py'),
  arguments: ['@INPUT@', '--config', config_tgt[1].full_path(), '-o', '@OUTPUT@'],
  output: '@BASENAME@',
  depends: config_tgt,
)

py.extension_module(
  'h5py.api_types_ext',
  template_gen.process('h5py/api_types_ext.pxd.template'),
  subdir: 'h5py',
  install: true,
  dependencies: npy_dep,
  c_args: npy_nodeprapi,
  limited_api: py.language_version(),
)

py.extension_module(
  'h5py.api_types_hdf5',
  template_gen.process('h5py/api_types_hdf5.pxd.template'),
  subdir: 'h5py',
  install: true,
  dependencies: npy_dep,
  c_args: npy_nodeprapi,
  limited_api: py.language_version(),
)

api_tgt = custom_target(
  'api',
  command: [
    find_program(helpers / 'api_gen.py'),
    '@SOURCE_ROOT@/h5py/api_functions.txt',
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

py.extension_module(
  'h5py.h5r',
  [template_gen.process('h5py/h5r.pxd.template'), 'h5py/h5r.pyx'],
  subdir: 'h5py',
  install: true,
  dependencies: [npy_dep, api_dep],
  c_args: npy_nodeprapi,
  limited_api: py.language_version(),
)

py.install_sources(
  'h5py/__init__.py',
  subdir: 'h5py',
)

hl = 'h5py/_hl'
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
