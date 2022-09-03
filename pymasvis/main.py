# -*- coding: utf-8 -*-
"""
Copyright 2012 Joakim Fors

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import collections
import locale
import logging
import operator
import os
import sys

from PIL import Image

from . import __version__
from .analysis import analyze
from .input import file_formats, load_file
from .output import render
from .utils import Steps, Timer

DEBUG = False


log = logging.getLogger('pymasvis')
lh = logging.StreamHandler(sys.stdout)
lh.setFormatter(logging.Formatter("%(message)s"))
log.addHandler(lh)
log.setLevel(logging.WARNING)


def run(
    infile,
    outfile=None,
    overviewfile=None,
    overviews=None,
    fmt='png',
    destdir='',
    update=True,
    header=None,
    r128_unit='LUFS',
):
    loader = None
    loader_args = []
    destdir = os.path.join(os.path.dirname(infile), destdir)
    if destdir and not os.path.isdir(destdir):
        log.debug("Creating destdir %s", destdir)
        os.mkdir(destdir)
    if os.path.isfile(infile):
        log.debug("Selecting file loader")
        loader = load_file
        loader_args = [infile]
        if not outfile:
            filename = os.path.basename(infile)
            filename = "%s-pymasvis.%s" % (filename, fmt)
            outfile = os.path.join(destdir, filename)
        if os.path.isfile(outfile):
            if update == 'no':
                log.warning("Destination file %s already exists", outfile)
                return
            elif update == 'outdated' and os.path.getmtime(outfile) > os.path.getmtime(
                infile
            ):
                log.warning(
                    "Destination file %s exist and is newer than %s", outfile, infile
                )
                return
        if not os.path.splitext(outfile)[1][1:] in ['png', 'jpg']:
            log.warning("Only png and jpg supported as output format")
            return
    else:
        log.warning("Unable to open input %s", infile)
        return
    track = loader(*loader_args)
    if type(track) is int:
        return
    if not header:
        header = "%s" % (track['metadata']['name'])
    with Timer('Running...', Steps.total, Steps.callback):
        with Timer('Analyzing...'):
            analysis = analyze(track, callback=Steps.callback)
        with Timer('Rendering...'):
            render_overview = False
            if overviewfile:
                render_overview = True
            detailed, overview = render(
                track,
                analysis,
                header,
                r128_unit=r128_unit,
                render_overview=render_overview,
                callback=Steps.callback,
            )
            img = Image.open(detailed)
            log.info("Writing %s", outfile)
            if fmt == 'png':
                img = img.convert(mode='P', palette='ADAPTIVE', colors=256)
                img.save(outfile, 'PNG', optimize=True)
            elif fmt == 'jpg':
                img.save(outfile, 'JPEG', quality=80, optimize=True)
            if overview:
                img_o = Image.open(overview)
                if overviewfile:
                    overviewindex = os.path.join(destdir, overviewfile)
                    if overviewindex not in overviews:
                        overviews[overviewindex] = []
                    overviews[overviewindex].append(img_o)
    Steps.report()


def main():
    import argparse
    from functools import reduce

    parser = argparse.ArgumentParser(description="Analyze audio file or Spotify URI.")
    parser.add_argument(
        '--version', action='version', version="PyMasVis " + __version__
    )
    parser.add_argument('-v', '--verbose', action='store_true', help="verbose messages")
    parser.add_argument('-d', '--debug', action='store_true', help="debug info")
    parser.add_argument('-u', '--username', metavar='username', help="Spotify username")
    parser.add_argument('-p', '--password', metavar='password', help="Spotify password")
    parser.add_argument(
        '-r',
        '--recursive',
        action='store_true',
        help="recurse directory tree if input is a directory",
    )
    parser.add_argument(
        '--destdir',
        metavar='destdir',
        type=str,
        default='',
        help="destination directory to store analysis in",
    )
    parser.add_argument(
        '--update',
        default='yes',
        type=str,
        choices=['yes', 'no', 'outdated'],
        help="choose wheter to update a result image or not, default: yes",
    )
    parser.add_argument(
        '--format',
        default='png',
        type=str,
        choices=['png', 'jpg'],
        help="selects output format, default: png",
    )
    parser.add_argument(
        '--overview',
        action='store_const',
        const='overview-pymasvis',
        help="generate overview",
    )
    parser.add_argument(
        '--overview-mode',
        default='dir',
        type=str,
        choices=['dir', 'flat'],
        help=(
            "generate an overview file per directory or one file for "
            "all inputs, default: dir"
        ),
    )
    parser.add_argument(
        '--lu',
        dest='r128_unit',
        default='LUFS',
        action='store_const',
        const='LU',
        help="Use LU instead of LUFS when displaying R128 values",
    )
    parser.add_argument(
        'inputs',
        metavar='input',
        type=str,
        nargs='+',
        help=(
            'a file, directory or Spotify URI (track, album or playlist) ' 'to analyze'
        ),
    )
    args = parser.parse_args()
    if args.verbose:
        log.setLevel(logging.INFO)
    if args.debug:
        log.setLevel(logging.DEBUG)
    if args.overview and args.update != 'yes':
        log.error("Update must be set to 'yes' to enable overviews")
        exit(1)
    if args.overview:
        args.overview += '.%s' % args.format
    formats = file_formats()
    infiles = []
    for f in args.inputs:
        if os.path.isfile(f):
            infiles.append(f)
            continue
        if os.path.isdir(f):
            for root, dirs, files in os.walk(f):
                for name in files:
                    if os.path.splitext(name)[1][1:] in formats:
                        infiles.append(os.path.join(root, name))
                if not args.recursive:
                    del dirs[:]
    if len(infiles) == 0:
        log.warning("No valid files for analysis found: " + " ".join(args.inputs))
    for fsenc in [sys.getfilesystemencoding(), locale.getdefaultlocale()[1]]:
        if fsenc:
            # encoding = fsenc
            break
    overviews = collections.OrderedDict()
    for infile in infiles:
        outfile = None
        header = None
        log.warning(infile)
        run(
            infile,
            outfile,
            args.overview,
            overviews,
            args.format,
            args.destdir,
            args.update,
            header,
            args.r128_unit,
        )
    if args.overview:
        if args.overview_mode == 'flat':
            if args.destdir:
                if not os.path.isdir(args.destdir):
                    log.debug("Creating destdir %s", args.destdir)
                    os.mkdir(args.destdir)
                args.overview = os.path.join(args.destdir, args.overview)
            overviews = {args.overview: reduce(operator.add, overviews.values())}
        for overviewfile, images in overviews.items():
            w, h = images[0].size
            n = len(images)
            out = Image.new('RGBA', (w, h * n))
            for i, image in enumerate(images):
                out.paste(image, (0, h * i))
            log.info("Writing overview %s", overviewfile)
            if args.format == 'png':
                out = out.convert(mode='P', palette='ADAPTIVE', colors=256)
                out.save(overviewfile, 'PNG', optimize=True)
            elif args.format == 'jpg':
                out.save(overviewfile, 'JPEG', quality=80, optimize=True)


if __name__ == "__main__":
    main()
