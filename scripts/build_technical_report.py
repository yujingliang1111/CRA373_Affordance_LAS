"""Build a standalone DOCX from the Traditional Chinese CRA373 report.

Uses only Python's standard library and Pillow (already present in the project
runtime). The Markdown file remains the editable report source.
"""
from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from io import BytesIO
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/CRA373_地形可供性技術說明.md'
OUTPUT = ROOT / 'docs/CRA373_地形可供性技術說明.docx'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
WP = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
PIC = 'http://schemas.openxmlformats.org/drawingml/2006/picture'
NS = f'xmlns:w="{W}" xmlns:r="{R}" xmlns:wp="{WP}" xmlns:a="{A}" xmlns:pic="{PIC}"'


def clean(value: str) -> str:
    value = value.strip()
    value = re.sub(r'\[([^]]+)\]\(([^)]+)\)', r'\1（\2）', value)
    return value.replace('**', '').replace('`', '')


def paragraph(value: str, style: str = 'Normal', *, page_break: bool = False,
              bold: bool = False, center: bool = False) -> str:
    ppr = f'<w:pStyle w:val="{style}"/>'
    if page_break:
        ppr += '<w:pageBreakBefore/>'
    if center:
        ppr += '<w:jc w:val="center"/>'
    rpr = '<w:rPr><w:b/></w:rPr>' if bold else ''
    return f'<w:p><w:pPr>{ppr}</w:pPr><w:r>{rpr}<w:t xml:space="preserve">{escape(clean(value))}</w:t></w:r></w:p>'


def table(rows: list[list[str]]) -> str:
    borders = ''.join(f'<w:{side} w:val="single" w:sz="4" w:color="B9C7CF"/>'
                      for side in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'))
    out = [f'<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/><w:tblBorders>{borders}</w:tblBorders>'
           '<w:tblCellMar><w:top w:w="85" w:type="dxa"/><w:left w:w="90" w:type="dxa"/>'
           '<w:bottom w:w="85" w:type="dxa"/><w:right w:w="90" w:type="dxa"/></w:tblCellMar>'
           '</w:tblPr>']
    for ri, row in enumerate(rows):
        out.append('<w:tr>')
        for item in row:
            fill = '<w:shd w:fill="DDE9ED"/>' if ri == 0 else ''
            out.append('<w:tc><w:tcPr>' + fill + '</w:tcPr>' +
                       paragraph(item, 'TableText', bold=(ri == 0)) + '</w:tc>')
        out.append('</w:tr>')
    out.append('</w:tbl>')
    return ''.join(out)


def image_paragraph(rid: str, doc_id: int, filename: str, cx: int, cy: int) -> str:
    return (f'<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:drawing>'
            f'<wp:inline distT="0" distB="0" distL="0" distR="0">'
            f'<wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{doc_id}" name="{escape(filename)}"/>'
            f'<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:pic><pic:nvPicPr><pic:cNvPr id="{doc_id}" name="{escape(filename)}"/>'
            f'<pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="{rid}"/>'
            f'<a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/>'
            f'</a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
            f'</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>')


def parse_report() -> tuple[str, list[tuple[str, bytes]]]:
    lines = SOURCE.read_text(encoding='utf-8').splitlines()
    body: list[str] = []
    media: list[tuple[str, bytes]] = []
    table_rows: list[list[str]] = []
    code = False
    next_image_id = 1

    def flush_table() -> None:
        if table_rows:
            body.append(table(table_rows.copy()))
            table_rows.clear()

    for raw in lines:
        line = raw.strip()
        if line.startswith('|') and line.endswith('|'):
            cells = [x.strip() for x in line.strip('|').split('|')]
            if all(re.fullmatch(r':?-+:?', x) for x in cells):
                continue
            table_rows.append(cells)
            continue
        flush_table()
        if line.startswith('```'):
            code = not code
            continue
        if code:
            body.append(paragraph(raw or ' ', 'Code'))
            continue
        if not line:
            continue
        if line.startswith('!['):
            match = re.fullmatch(r'!\[([^]]*)\]\(([^)]+)\)', line)
            if not match:
                raise ValueError(f'Invalid report image syntax: {line}')
            caption, relative = match.groups()
            img_path = (SOURCE.parent / relative).resolve()
            if not img_path.is_file() or ROOT not in img_path.parents:
                raise FileNotFoundError(img_path)
            im = Image.open(img_path)
            # A4 printable area: keep full dashboard screenshots within one page.
            max_width_in = 6.0 if 'isaac' in img_path.name else 4.65
            max_height_in = 8.25
            scale = min(max_width_in / im.width, max_height_in / im.height)
            cx = round(im.width * scale * 914400)
            cy = round(im.height * scale * 914400)
            rid = f'rIdImage{next_image_id}'
            media.append((img_path.name, img_path.read_bytes()))
            body.append(image_paragraph(rid, next_image_id, img_path.name, cx, cy))
            body.append(paragraph(f'圖：{caption}', 'Caption', center=True))
            next_image_id += 1
            continue
        if line.startswith('# '):
            body.append(paragraph(line[2:], 'Title', center=True))
        elif line.startswith('## '):
            heading = line[3:]
            body.append(paragraph(heading, 'Heading1', page_break=(heading == '摘要')))
        elif line.startswith('### '):
            heading = line[4:]
            body.append(paragraph(heading, 'Heading2', page_break=heading.startswith('圖 ')))
        elif re.match(r'^\d+\. ', line):
            body.append(paragraph(line, 'List'))
        elif line.startswith('- '):
            body.append(paragraph('• ' + line[2:], 'List'))
        else:
            body.append(paragraph(line.replace('  ', ' '), 'Normal'))
    flush_table()
    body.append('<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
                '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134"/>'
                '</w:sectPr>')
    return ''.join(body), media


def main() -> None:
    content, media = parse_report()
    document = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' \
        f'<w:document {NS}><w:body>{content}</w:body></w:document>'
    styles = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <w:styles xmlns:w="{W}">
    <w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Noto Sans CJK TC" w:hAnsi="Noto Sans CJK TC" w:eastAsia="Noto Sans CJK TC"/><w:sz w:val="20"/></w:rPr></w:rPrDefault>
    <w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="300" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>
    <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
    <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:pPr><w:spacing w:before="900" w:after="520"/></w:pPr><w:rPr><w:b/><w:sz w:val="36"/><w:color w:val="183849"/></w:rPr></w:style>
    <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:pPr><w:spacing w:before="330" w:after="170"/><w:keepNext/></w:pPr><w:rPr><w:b/><w:sz w:val="29"/><w:color w:val="164C65"/></w:rPr></w:style>
    <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:pPr><w:spacing w:before="260" w:after="140"/><w:keepNext/></w:pPr><w:rPr><w:b/><w:sz w:val="24"/><w:color w:val="245A6E"/></w:rPr></w:style>
    <w:style w:type="paragraph" w:styleId="TableText"><w:name w:val="Table Text"/><w:pPr><w:spacing w:after="0"/></w:pPr><w:rPr><w:sz w:val="17"/></w:rPr></w:style>
    <w:style w:type="paragraph" w:styleId="Code"><w:name w:val="Code"/><w:pPr><w:spacing w:after="0"/></w:pPr><w:rPr><w:rFonts w:ascii="DejaVu Sans Mono" w:hAnsi="DejaVu Sans Mono"/><w:sz w:val="16"/></w:rPr></w:style>
    <w:style w:type="paragraph" w:styleId="Caption"><w:name w:val="Caption"/><w:pPr><w:spacing w:after="180"/></w:pPr><w:rPr><w:i/><w:sz w:val="17"/><w:color w:val="506873"/></w:rPr></w:style>
    <w:style w:type="paragraph" w:styleId="List"><w:name w:val="List"/><w:pPr><w:ind w:left="360"/><w:spacing w:after="80"/></w:pPr></w:style>
    </w:styles>'''
    rels = ''.join(f'<Relationship Id="rIdImage{i}" '
                   'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
                   f'Target="media/{escape(name)}"/>' for i, (name, _) in enumerate(media, 1))
    rels += '<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + rels + '</Relationships>'
    types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/><Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/></Types>'''
    root_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/></Relationships>'''
    now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    core = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>CRA373 視覺化地形可供性與潛在動作空間技術說明</dc:title><dc:creator>CRA373_Affordance_LAS</dc:creator><dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created></cp:coreProperties>'''
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(OUTPUT, 'w', ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', types)
        z.writestr('_rels/.rels', root_rels)
        z.writestr('docProps/core.xml', core)
        z.writestr('word/document.xml', document)
        z.writestr('word/styles.xml', styles)
        z.writestr('word/_rels/document.xml.rels', rels)
        for name, data in media:
            z.writestr('word/media/' + name, data)
    print(f'{OUTPUT} ({OUTPUT.stat().st_size:,} bytes; {len(media)} figures)')


if __name__ == '__main__':
    main()
