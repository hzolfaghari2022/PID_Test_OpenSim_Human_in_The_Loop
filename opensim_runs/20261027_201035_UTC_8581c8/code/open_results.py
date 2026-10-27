"""Build and open a local figure gallery; no OpenSim or plotting imports needed."""
import argparse
from html import escape
from pathlib import Path
import os
from urllib.parse import quote
import webbrowser

ROOT = Path(__file__).resolve().parent

def build_gallery(suite):
    suite = Path(suite).resolve()
    if not suite.is_dir():
        raise FileNotFoundError(suite)
    pictures = sorted(suite.glob('*.png')) + sorted(suite.glob('*/tracking.png'))
    if not pictures:
        raise ValueError('No saved figures found in ' + str(suite))
    cards = []
    for picture in pictures:
        relative = picture.relative_to(suite).as_posix()
        url = quote(relative)
        title = escape(relative.replace('_', ' '))
        pdf = picture.with_suffix('.pdf')
        pdf_link = (' | <a href="' + quote(pdf.relative_to(suite).as_posix()) + '">Open PDF</a>') if pdf.exists() else ''
        cards.append(f'<section><h2>{title}</h2><p><a href="{url}" target="_blank">Open full-size figure</a>{pdf_link}</p><a href="{url}" target="_blank"><img src="{url}" alt="{title}" loading="lazy"></a></section>')
    answer = suite / 'ANSWER_READ_FIRST.txt'
    explanation = '<pre>' + escape(answer.read_text(encoding='utf-8')) + '</pre>' if answer.exists() else ''
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Arm26 simulation results</title>
<style>body{font:18px Georgia,serif;max-width:1200px;margin:32px auto;padding:0 24px;background:#f3f5f7;color:#17212b}section{background:white;padding:20px;margin:24px 0;border:1px solid #ddd;border-radius:8px}img{width:100%;height:auto}a{color:#125ab0}pre{white-space:pre-wrap;font:16px/1.5 Georgia,serif}h2{font-size:21px}</style>
<h1>Arm26 simulation results</h1><p>Click a figure to open it at full size. Use your browser zoom to inspect details.</p>'''
    document += '<p>Run: ' + escape(suite.name) + '</p>' + explanation + ''.join(cards) + '</html>'
    gallery = suite / 'VIEW_RESULTS.html'
    gallery.write_text(document, encoding='utf-8')
    return gallery

def open_gallery(suite):
    gallery = build_gallery(suite)
    print('FIGURE GALLERY:', gallery)
    try:
        if os.name == 'nt':
            os.startfile(str(gallery))
            return True
        opened = webbrowser.open(gallery.as_uri(), new=2)
        if not opened:
            print('No browser opened. Double-click VIEW_RESULTS.html in the results folder.')
        return bool(opened)
    except Exception as exc:
        print('Could not open the browser:', exc)
        print('Double-click VIEW_RESULTS.html in the results folder.')
        return False

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite', nargs='?', type=Path, help='Results folder; default is latest completed suite.')
    parser.add_argument('--no-open', action='store_true', help='Build gallery only.')
    args = parser.parse_args()
    suite = args.suite
    if suite is None:
        candidates = [p for p in (ROOT / 'speed_test_results').glob('*')
                      if p.is_dir() and (p/'summary.json').exists() and list(p.glob('*.png'))
                      and not (p/'INTERRUPTED_FOR_RUNTIME_CONFIGURATION.txt').exists()]
        if not candidates:
            parser.error('No saved results yet. Run a simulation first.')
        suite = max(candidates, key=lambda p: p.name)
    if args.no_open:
        print(build_gallery(suite))
    else:
        open_gallery(suite)

if __name__ == '__main__':
    main()
