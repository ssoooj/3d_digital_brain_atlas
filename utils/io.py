# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""IO utilities."""

import errno
import glob
import hashlib
import logging
import os
import pickle
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import urllib
import urllib.request
import zipfile
from uuid import uuid4
from pathlib import PurePath

from pkg_resources import parse_version

logger = logging.getLogger(__name__)

_DETECTRON_S3_BASE_URL = 'https://dl.fbaipublicfiles.com/detectron'


def save_object(obj, file_name, pickle_format=pickle.HIGHEST_PROTOCOL):
    """Save a Python object by pickling it.

Unless specifically overridden, we want to save it in Pickle format=2 since this
will allow other Python2 executables to load the resulting Pickle. When we want
to completely remove Python2 backward-compatibility, we can bump it up to 3. We
should never use pickle.HIGHEST_PROTOCOL as far as possible if the resulting
file is manifested or used, external to the system.
    """
    file_name = os.path.abspath(file_name)
    # Avoid filesystem race conditions (particularly on network filesystems)
    # by saving to a random tmp file on the same filesystem, and then
    # atomically rename to the target filename.
    tmp_file_name = file_name + ".tmp." + uuid4().hex
    try:
        with open(tmp_file_name, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle_format)
            f.flush()  # make sure it's written to disk
            os.fsync(f.fileno())
        os.rename(tmp_file_name, file_name)
    finally:
        # Clean up the temp file on failure. Rather than using os.path.exists(),
        # which can be unreliable on network filesystems, attempt to delete and
        # ignore os errors.
        try:
            os.remove(tmp_file_name)
        except EnvironmentError as e:  # parent class of IOError, OSError
            if getattr(e, 'errno', None) != errno.ENOENT:  # We expect ENOENT
                logger.info("Could not delete temp file %r",
                            tmp_file_name, exc_info=True)
                # pass through since we don't want the job to crash


def load_object(file_name):
    with open(file_name, 'rb') as f:
        # The default encoding used while unpickling is 7-bit (ASCII.) However,
        # the blobs are arbitrary 8-bit bytes which don't agree. The absolute
        # correct way to do this is to use `encoding="bytes"` and then interpret
        # the blob names either as ASCII, or better, as unicode utf-8. A
        # reasonable fix, however, is to treat it the encoding as 8-bit latin1
        # (which agrees with the first 256 characters of Unicode anyway.)
        return pickle.load(f, encoding='latin1')


def cache_url(url_or_file, cache_dir):
    """Download the file specified by the URL to the cache_dir and return the
    path to the cached file. If the argument is not a URL, simply return it as
    is.
    """
    is_url = re.match(
        r'^(?:http)s?://', url_or_file, re.IGNORECASE
    ) is not None

    if not is_url:
        return url_or_file

    url = url_or_file
    assert url.startswith(_DETECTRON_S3_BASE_URL), \
        ('Detectron only automatically caches URLs in the Detectron S3 '
         'bucket: {}').format(_DETECTRON_S3_BASE_URL)

    cache_file_path = url.replace(_DETECTRON_S3_BASE_URL, cache_dir)
    if os.path.exists(cache_file_path):
        assert_cache_file_is_ok(url, cache_file_path)
        return cache_file_path

    cache_file_dir = os.path.dirname(cache_file_path)
    if not os.path.exists(cache_file_dir):
        os.makedirs(cache_file_dir)

    logger.info('Downloading remote file {} to {}'.format(url, cache_file_path))
    download_url(url, cache_file_path)
    assert_cache_file_is_ok(url, cache_file_path)
    return cache_file_path


def assert_cache_file_is_ok(url, file_path):
    """Check that cache file has the correct hash."""
    # File is already in the cache, verify that the md5sum matches and
    # return local path
    cache_file_md5sum = _get_file_md5sum(file_path)
    ref_md5sum = _get_reference_md5sum(url)
    assert cache_file_md5sum == ref_md5sum, \
        ('Target URL {} appears to be downloaded to the local cache file '
         '{}, but the md5 hash of the local file does not match the '
         'reference (actual: {} vs. expected: {}). You may wish to delete '
         'the cached file and try again to trigger automatic '
         'download.').format(url, file_path, cache_file_md5sum, ref_md5sum)


def _progress_bar(count, total):
    """Report download progress.
    Credit:
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write(
        '  [{}] {}% of {:.1f}MB file  \r'.
            format(bar, percents, total / 1024 / 1024)
    )
    sys.stdout.flush()
    if count >= total:
        sys.stdout.write('\n')


def download_url(
        url, dst_file_path, chunk_size=8192, progress_hook=_progress_bar
):
    """Download url and write it to dst_file_path.
    Credit:
    https://stackoverflow.com/questions/2028517/python-urllib2-progress-hook
    """
    response = urllib.request.urlopen(url)
    total_size = response.info().get('Content-Length').strip()
    total_size = int(total_size)
    bytes_so_far = 0

    with open(dst_file_path, 'wb') as f:
        while 1:
            chunk = response.read(chunk_size)
            bytes_so_far += len(chunk)
            if not chunk:
                break
            if progress_hook:
                progress_hook(bytes_so_far, total_size)
            f.write(chunk)

    return bytes_so_far


def _get_file_md5sum(file_name):
    """Compute the md5 hash of a file."""
    hash_obj = hashlib.md5()
    with open(file_name, 'rb') as f:
        hash_obj.update(f.read())
    return hash_obj.hexdigest().encode('utf-8')


def _get_reference_md5sum(url):
    """By convention the md5 hash for url is stored in url + '.md5sum'."""
    url_md5sum = url + '.md5sum'
    md5sum = urllib.request.urlopen(url_md5sum).read().strip()
    return md5sum


def change_folder_permissions_recursive(path: str, mode=stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR):
    for root, dirs, files in os.walk(path, topdown=False):
        for fd in dirs:
            os.chmod(os.path.join(root, fd), mode)
        for file in files:
            os.chmod(os.path.join(root, file), mode)
    os.chmod(path, mode)


def jinny_nas_dir() -> str:
    if is_linux():
        # return '/run/user/1001/gvfs/afp-volume:host=DiskStation.local,user=jinny_nas,volume=shared'
        return '/media/jinny_nas'
    elif is_mac():
        return '/Volumes/shared'
    else:
        raise Exception('not supported')


def fs3017_dir() -> str:
    if is_linux():
        return '/media/fs3017'
    elif is_mac():
        return '/Volumes/fs3017'
    else:
        raise Exception('not supported')


def fs3017_data_dir() -> str:
    if is_linux():
        return '/media/fs3017_data'
    elif is_mac():
        return '/Volumes/fs3017_data'
    else:
        raise Exception('not supported')


def use_ninja() -> bool:
    return True


def curr_dir() -> str:
    return os.path.abspath(os.path.dirname(__file__))


def atlas_repository_dir() -> str:
    res = os.path.normpath(os.path.join(curr_dir(), '..'))
    assert os.path.exists(res)
    return res


def base_dir() -> str:
    res = os.path.normpath(os.path.join(atlas_repository_dir(), '..'))
    assert os.path.exists(res)
    return res


def atlas_src_dir() -> str:
    res = os.path.join(atlas_repository_dir(), 'src')
    assert os.path.exists(res)
    return res


def ext_dir() -> str:
    res = os.path.join(atlas_src_dir(), '3rdparty')
    assert os.path.exists(res)
    return res


def atlas_dir() -> str:
    res = os.path.join(atlas_src_dir(), 'atlas')
    assert os.path.exists(res)
    return res


def img_dir() -> str:
    res = os.path.join(atlas_src_dir(), 'img')
    assert os.path.exists(res)
    return res


def python_package_dir() -> str:
    res = os.path.join(atlas_src_dir(), 'python')
    assert os.path.exists(res)
    return res


def resource_dir() -> str:
    res = os.path.join(atlas_dir(), 'Resources')
    assert os.path.exists(res)
    return res


def src_package_dir() -> str:
    res = os.path.join(base_dir(), 'atlas_others')
    if not os.path.exists(res):
        if sys.platform.startswith('win'):
            res = os.path.join('Z:', os.sep, 'Google Drive', 'code', 'my', 'atlas_others')
        else:
            res = os.path.join(os.path.expanduser('~'), 'Google Drive', 'code', 'my', 'atlas_others')
    assert os.path.exists(res)
    return res


def atlas_build_dir() -> str:
    if use_ninja():
        res = os.path.join(atlas_repository_dir(), 'cmake-build-release-ninja')
    else:
        res = os.path.join(atlas_repository_dir(), 'cmake-build-release')
    if not os.path.exists(res):
        os.mkdir(res)
    assert os.path.exists(res)
    return res


def python_package_build_dir() -> str:
    if use_ninja():
        res = os.path.join(atlas_repository_dir(), 'cmake-build-python-ninja')
    else:
        res = os.path.join(atlas_repository_dir(), 'cmake-build-python')
    if not os.path.exists(res):
        os.mkdir(res)
    assert os.path.exists(res)
    return res


def atlas_binary_dir() -> str:
    res = os.path.join(atlas_build_dir(), 'src', 'atlas')
    if not use_ninja() and sys.platform.startswith('win32'):
        res = os.path.join(atlas_build_dir(), 'src', 'atlas', 'Release')
    assert os.path.exists(res)
    return res


def deploy_target_dir() -> str:
    return os.path.join(atlas_repository_dir(), 'deploy')


def qt_install_dir() -> str:
    if sys.platform.startswith('win32'):
        res = os.path.join('C:', os.sep, 'Qt')
    elif sys.platform.startswith('darwin'):
        res = os.path.join(os.path.expanduser('~'), 'Qt')
    else:
        res = os.path.join(os.path.expanduser('~'), 'Qt')
    assert os.path.exists(res)
    return res


def qt_compiler_name() -> str:
    if sys.platform.startswith('win32'):
        return 'msvc2017_64'
    elif sys.platform.startswith('darwin'):
        return 'clang_64'
    else:
        return 'gcc_64'


def qmake_bin_name() -> str:
    if sys.platform.startswith('win32'):
        return 'qmake.exe'
    else:
        return 'qmake'


def qt_ver() -> str:
    vers = [fd for fd in os.listdir(qt_install_dir()) if
            os.path.exists(os.path.join(qt_install_dir(), fd, qt_compiler_name(), 'bin', qmake_bin_name()))]
    assert vers
    vers = sorted(vers, key=parse_version)
    ver = vers[-1]
    return ver


def qt_base_dir() -> str:
    return os.path.join(qt_install_dir(), qt_ver(), qt_compiler_name())


def qt_bin_dir() -> str:
    return os.path.join(qt_base_dir(), 'bin')


def qmake_bin() -> str:
    return os.path.join(qt_bin_dir(), qmake_bin_name())


def qt_installer_framework_ver() -> str:
    folder = os.path.join(qt_install_dir(), 'Tools', 'QtInstallerFramework')
    vers = [fd for fd in os.listdir(folder) if
            os.path.exists(os.path.join(folder, fd, 'bin'))]
    assert vers
    vers = sorted(vers, key=parse_version)
    ver = vers[-1]
    return ver


def qt_installer_framework_bin_dir() -> str:
    folder = os.path.join(qt_install_dir(), 'Tools', 'QtInstallerFramework')
    return os.path.join(folder, qt_installer_framework_ver(), 'bin')


def vs_install_dir() -> str:
    assert sys.platform.startswith('win32')

    vsinstalldir = r'C:\Program Files (x86)\Microsoft Visual Studio\2019\Community'
    assert os.path.exists(vsinstalldir)

    return vsinstalldir


def vc_redist_dir() -> str:
    assert sys.platform.startswith('win32')

    vc_redist_version_filename = os.path.join(vs_install_dir(), 'VC', 'Auxiliary', 'Build',
                                              'Microsoft.VCRedistVersion.default.txt')
    assert os.path.exists(vc_redist_version_filename)
    with open(vc_redist_version_filename, mode='r', encoding='utf-8') as f:
        vc_redist_version = f.readline().rstrip()

    res = os.path.join(vs_install_dir(), 'VC', 'Redist', 'MSVC', vc_redist_version)
    assert os.path.exists(res)
    return res


def vc_CRT_redist_dir() -> str:
    assert sys.platform.startswith('win32')

    res = os.path.join(vc_redist_dir(), 'x64', 'Microsoft.VC141.CRT')
    assert os.path.exists(res)
    return res


def vc_CXXAMP_redist_dir() -> str:
    assert sys.platform.startswith('win32')

    res = os.path.join(vc_redist_dir(), 'x64', 'Microsoft.VC141.CXXAMP')
    assert os.path.exists(res)
    return res


def vc_OpenMP_redist_dir() -> str:
    assert sys.platform.startswith('win32')

    res = os.path.join(vc_redist_dir(), 'x64', 'Microsoft.VC141.OpenMP')
    assert os.path.exists(res)
    return res


def intel_sw_dir() -> str:
    if sys.platform.startswith('win32'):
        res = os.path.join('C:', os.sep, 'Program Files (x86)', 'IntelSWTools', 'compilers_and_libraries', 'windows')
    else:
        res = os.path.join(os.sep, 'opt', 'intel')
    assert os.path.exists(res)
    return res


def tbb_redist_dir() -> str:
    assert sys.platform.startswith('win32')

    res = os.path.join('C:', os.sep, 'Program Files (x86)', 'IntelSWTools', 'compilers_and_libraries',
                       'windows', 'redist', 'intel64', 'tbb', 'vc14')
    assert os.path.exists(res)
    return res


def assimp_redist_dir() -> str:
    assert sys.platform.startswith('win32')

    res = os.path.join(ext_dir(), 'assimp', 'bin')
    assert os.path.exists(res)
    return res


def freeimage_redist_dir() -> str:
    assert sys.platform.startswith('win32')

    res = os.path.join(ext_dir(), 'freeimage')
    assert os.path.exists(res)
    return res


def software_dir() -> str:
    res = os.path.join(os.path.expanduser('~'), 'software')
    if not os.path.exists(res):
        os.mkdir(res)
    assert os.path.exists(res)
    return res


def is_windows() -> bool:
    return sys.platform.startswith('win')


def is_mac() -> bool:
    return sys.platform.startswith('darwin')


def is_linux() -> bool:
    return sys.platform.startswith('linux')


def find_src_package_with_glob(files: str):
    file_list = glob.glob(files)
    if len(file_list) == 1:
        return file_list[0]
    elif len(file_list) == 0:
        raise Exception("Can not find matching package with pattern: " + files)
    else:
        raise Exception("Find more than one matching packages with pattern: " + files)


def get_package_top_level_folder(file: str, folder: str):
    res = ''
    if file.lower().endswith('.zip'):
        with zipfile.ZipFile(file, mode='r') as zf:
            # logger.info(zf.namelist())
            res = os.path.join(folder, os.path.commonpath([nm for nm in zf.namelist() if not nm.endswith('/')]))
    elif file.lower().endswith('.tar.gz') or file.lower().endswith('.tar.bz2') or file.lower().endswith('.tar.xz') \
            or file.lower().endswith('.tgz'):
        with tarfile.open(file, mode='r|*') as tf:
            names = [nm for nm in tf.getnames() if not nm == '.']
            res = os.path.join(folder, os.path.commonpath(names))
    elif file.lower().endswith('.7z'):
        cp = subprocess.run(['7za', 'l', '-slt', file], stdout=subprocess.PIPE, encoding='utf-8')
        started = False
        filenames = []
        for line in cp.stdout.splitlines():
            if started:
                if line.startswith('Path = '):
                    filenames.append(line.replace('Path = ', ''))
            else:
                if line.startswith('-------'):
                    started = True
        res = os.path.join(folder, os.path.commonpath(filenames))

    if res.endswith('/') or res.endswith('\\'):
        res = res[:-1]
    return res


def unpack_file_to_folder(file: str, folder: str):
    logger.info(f'unpacking {file}')
    if file.lower().endswith('.zip'):
        with zipfile.ZipFile(file, mode='r') as zf:
            zf.extractall(path=folder)
    elif file.lower().endswith('.tar.gz') or file.lower().endswith('.tar.bz2') or file.lower().endswith('.tar.xz') \
            or file.lower().endswith('.tgz'):
        with tarfile.open(file, mode='r|*') as tf:
            tf.extractall(path=folder)
    elif file.lower().endswith('.7z'):
        if is_windows():
            subprocess.run(['7za', 'x', '-y', '-o' + folder, file],
                           shell=False, check=True, cwd=curr_dir())
        else:
            subprocess.run(['7za', 'x', '-y', '-o' + folder, file],
                           shell=False, check=True)


def unpack_tool_to_software_dir(tool_package_folder: str, tool_package_glob_name: str,
                                tool_folder_glob_name=None) -> str:
    if tool_folder_glob_name is None:
        tool_folder_glob_name = tool_package_glob_name
    package_name = find_src_package_with_glob(os.path.join(tool_package_folder, tool_package_glob_name))
    package_unpack_folder = get_package_top_level_folder(package_name, software_dir())
    if not os.path.exists(package_unpack_folder):
        folder_list = glob.glob(os.path.join(software_dir(), tool_folder_glob_name))
        if len(folder_list) == 1:
            shutil.rmtree(folder_list[0], ignore_errors=False)
        unpack_file_to_folder(package_name, software_dir())
    return package_unpack_folder


def install_cmake():
    if is_windows():
        unpack_tool_to_software_dir(src_package_dir(), 'cmake*win64*')
    elif is_linux():
        unpack_tool_to_software_dir(src_package_dir(), 'cmake*Linux*')
    else:
        unpack_tool_to_software_dir(src_package_dir(), 'cmake*Darwin*')


def install_ninja():
    if is_windows():
        unpack_file_to_folder(os.path.join(src_package_dir(), 'ninja-win.zip'), software_dir())
    elif is_linux():
        if os.path.exists(os.path.join(software_dir(), 'ninja')):
            os.remove(os.path.join(software_dir(), 'ninja'))
        unpack_file_to_folder(os.path.join(src_package_dir(), 'ninja-linux.zip'), software_dir())
        os.chmod(os.path.join(software_dir(), 'ninja'), stat.S_IXUSR)
    else:
        if os.path.exists(os.path.join(software_dir(), 'ninja')):
            os.remove(os.path.join(software_dir(), 'ninja'))
        unpack_file_to_folder(os.path.join(src_package_dir(), 'ninja-mac.zip'), software_dir())
        os.chmod(os.path.join(software_dir(), 'ninja'), stat.S_IXUSR)


def install_ffmpeg():
    if is_windows():
        unpack_tool_to_software_dir(src_package_dir(), 'ffmpeg*win*')
    elif is_linux():
        unpack_tool_to_software_dir(src_package_dir(), 'ffmpeg*static.tar.xz')
    else:
        folder = unpack_tool_to_software_dir(src_package_dir(), 'ffmpeg*macos*')
        os.chmod(os.path.join(folder, 'bin', 'ffmpeg'), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        os.chmod(os.path.join(folder, 'bin', 'ffplay'), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        os.chmod(os.path.join(folder, 'bin', 'ffprobe'), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        os.symlink(os.path.join(folder, 'bin', 'ffmpeg'), '/usr/local/bin/ffmpeg')


def get_cmake_binary() -> str:
    if is_windows():
        cmake_folder = find_src_package_with_glob(os.path.join(software_dir(), 'cmake-*win*-x64'))
        return os.path.join(cmake_folder, 'bin', 'cmake')
    elif is_linux():
        cmake_folder = find_src_package_with_glob(os.path.join(software_dir(), 'cmake-*Linux*_64'))
        return os.path.join(cmake_folder, 'bin', 'cmake')
    else:
        cmake_folder = find_src_package_with_glob(os.path.join(software_dir(), 'cmake-*Darwin*_64'))
        return os.path.join(cmake_folder, 'CMake.app', 'Contents', 'bin', 'cmake')


def get_ninja_binary() -> str:
    if is_windows():
        return os.path.join(software_dir(), 'ninja.exe')
    else:
        return os.path.join(software_dir(), 'ninja')


def get_ffmpeg_binary() -> str:
    if is_windows():
        folder = find_src_package_with_glob(os.path.join(software_dir(), 'ffmpeg*win*'))
        return os.path.join(folder, 'bin', 'ffmpeg.exe')
    elif is_linux():
        folder = find_src_package_with_glob(os.path.join(software_dir(), 'ffmpeg*static'))
        return os.path.join(folder, 'ffmpeg')
    else:
        folder = find_src_package_with_glob(os.path.join(software_dir(), 'ffmpeg*macos*'))
        return os.path.join(folder, 'bin', 'ffmpeg')


def update_or_clone_git_repository(repository_folder: str, repository_url: str):
    if os.path.exists(repository_folder):
        print('git', 'pull', PurePath(repository_folder).name)
        subprocess.run(['git', 'pull'], cwd=repository_folder, shell=False, check=False)
    else:
        subprocess.run(['git', 'clone', repository_url, repository_folder], shell=False, check=True)


def update_or_checkout_svn_repository(repository_folder: str, repository_url: str):
    if os.path.exists(repository_folder):
        print('svn', 'up', PurePath(repository_folder).name)
        subprocess.run(['svn', 'up'], cwd=repository_folder, shell=False, check=False)
    else:
        subprocess.run(['svn', 'checkout', repository_url, repository_folder], shell=False, check=True)


def update_or_clone_git_repository_with_submodules(repository_folder: str, repository_url: str):
    if os.path.exists(repository_folder):
        print('git', 'pull', PurePath(repository_folder).name)
        subprocess.run(['git', 'pull'], cwd=repository_folder, shell=False, check=False)
        subprocess.run(['git', 'submodule', 'update', '--init'], cwd=repository_folder, shell=False, check=False)
    else:
        subprocess.run(['git', 'clone', '--recursive', repository_url, repository_folder], shell=False, check=True)


def export_git_repository(repository_folder: str, target_folder: str, branch: str = '', tag: str = ''):
    if not branch:
        branch = 'master'
    shutil.rmtree(target_folder, ignore_errors=True)
    subprocess.run(['git', 'clone', '--shared', '--branch', branch, repository_folder, target_folder],
                   shell=False, check=True)
    if tag:
        subprocess.run(['git', 'checkout', tag], cwd=target_folder, shell=False, check=True)
    shutil.rmtree(os.path.join(target_folder, '.git'), ignore_errors=False)


def code_dir() -> str:
    res = os.path.join(os.path.expanduser('~'), 'code')
    if not os.path.exists(res):
        os.mkdir(res)
    assert os.path.exists(res)
    return res


if __name__ == "__main__":
    logger.info('nothing')
