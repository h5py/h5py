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
template_subs = custom_target(
  'template_subs',
  command: [find_program(helpers / 'config.py'), '-o', '@OUTDIR@'],
  output: 'template_substitutions.json',
)

template = generator(
  find_program(helpers / 'template.py'),
  arguments: ['@INPUT@', '--config', template_subs.full_path(), '-o', '@OUTPUT@'],
  output: '@BASENAME@',
  depends: template_subs,
)

py.extension_module(
    'h5py.h5r',
    [template.process('h5py/h5r.pxd.template'), 'h5py/h5r.pyx'],
    subdir: 'h5py',
    install: true,
    dependencies : npy_dep,
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
