import sys
import requests
import os
import click
import traceback
import tarfile
import zipfile


def _cerr(message: str):
    click.echo(click.style(f" {message}", fg = 'bright_red'))


def cerr(*messages):
    val = ''
    for message in messages:
        val = val + ' ' + str(message)
    _cerr(val)


def _download_file(url, folder, headers):
    os.makedirs(folder, exist_ok=True)
    fname = os.path.join(folder, url.split('?')[0].split('/')[-1])
    print(f'start downloading: {url} => {fname}')
    ret = fname
    try:
        # start and block request
        r = requests.get(url, stream=True, headers=headers, timeout=3000)
        # obtain content length
        length = int(r.headers['content-length'])
        print(f'file size: {size_description(length)}')
        if os.path.exists(fname) and os.path.getsize(fname) == length:
            cerr(f'file already exists {fname}')
            return ret
        # start writing
        f = open(fname, 'wb+')
        # show in progressbar
        with click.progressbar(label="Downloading from remote: ", length=length) as bar:
            for chunk in r.iter_content(chunk_size = 512):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        print('Download Complete.')
        f.close()
    except Exception as err:
        cerr(f'Error: {repr(err)}')
        traceback.print_exc()
        ret = 1
    finally:
        return ret


def download_file(url, folder, headers={}, trial=5):
    fail_count = 0
    while True:
        ret = _download_file(url, folder, headers)
        if ret != 1:
            return ret
        if fail_count < trial:
            fail_count += 1
            cerr(f'Download failed, Trial {fail_count}/{trial}')
        else:
            cerr('Download failed. Exceeded trial limit.')
            return -1


def download_file_extract(url, folder, headers={}, trial=5):
    if download_file(url, folder, headers, trial) < 0: return
    fname = os.path.join(folder, url.split('?')[0].split('/')[-1])
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False
    fp.extractall(folder)
    return os.path.join(folder, data_dir)


def _download_webpage(url, headers, encoding):
    '''
    Download webpage from url.
    :param url: url to download
    '''
    print(f'start downloading: {url} => memory')
    # download
    try:
        return requests.get(url=url, headers=headers).content.decode(encoding)
    except Exception as e:
        cerr(f'error: {repr(e)}')
        traceback.print_exc()
        return -1


def download_webpage(url, headers, encoding='utf-8', trial=5):
    '''
    Download webpage from url.
    :param url: url to download
    :param trial: number of trials
    '''
    fail_count = 0
    while True:
        ret = _download_webpage(url, headers, encoding)
        if ret != -1:
            return ret
        if fail_count < trial:
            fail_count += 1
            cerr(f'Download failed, Trial {fail_count}/{trial}')
        else:
            cerr('Download failed. Exceeded trial limit.')


def size_description(size):
    '''
    Taken and modified from https://blog.csdn.net/wskzgz/article/details/99293181
    '''
    def strofsize(integer, remainder, level):
        if integer >= 1024:
            remainder = integer % 1024
            integer //= 1024
            level += 1
            return strofsize(integer, remainder, level)
        else:
            return integer, remainder, level

    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    integer, remainder, level = strofsize(size, 0, 0)
    if level + 1 > len(units):
        level = -1
    return ( '{}.{:>03d} {}'.format(integer, remainder, units[level]) )
