"""Generate a pkgconfig file for the library."""
import argparse


def print_pkgconf(prefix, lib_name):
    print(f'prefix={prefix}')
    print("""exec_prefix=${prefix}
libdir=${exec_prefix}/lib
includedir=${prefix}/include

Name: sparse_tensor
Description: sparse tensor library
Version: 0.0.1""")
    print()
    print('Requires:')
    print('Libs: -L${libdir} -l%s -Wl,-rpath,${libdir}' % (lib_name,))
    print('Cflags: -I${includedir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix')
    parser.add_argument('lib_name')
    args = parser.parse_args()
    print_pkgconf(prefix=args.prefix, lib_name=args.lib_name)
