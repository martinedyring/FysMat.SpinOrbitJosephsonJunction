CFLAGS=-stdlib=libc++

if platform.system() == 'Darwin':
    extra_compile_args += ['-mmacosx-version-min=10.7', '-stdlib=libc++']

export MACOSX_DEPLOYMENT_TARGET=10.8