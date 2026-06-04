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
template_prog = find_program(helpers / 'template.py')
template_base_cmd = [template_prog, '--config', config_tgt[1].full_path(), '-o', '@OUTPUT@']

subdir('h5py')