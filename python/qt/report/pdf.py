"""
File: pdf.py
Author: Chuncheng Zhang
Date: 2025-04-01
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Generate PDF file by reportlab.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-04-01 ------------------------
# Requirements and constants
import os
import matplotlib.pyplot as plt

from io import BytesIO
from datetime import datetime
from PIL import Image as PILImage

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, BaseDocTemplate, Frame, PageTemplate
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

# %% ---- 2025-04-01 ------------------------
# Function and class


class PDFReportGenerator:
    def __init__(self, title, output_path="report.pdf", page_size=A4):
        """
        初始化PDF报告生成器

        参数:
            title: 报告标题
            output_path: 输出PDF路径 (默认: "report.pdf")
            page_size: 页面尺寸 (默认: A4)
        """
        self.title = title
        self.output_path = output_path
        self.page_size = page_size
        self.elements = []
        self.chinese_font = self._register_chinese_font()
        self.styles = self._setup_styles()
        self.count_pages = 0
        self.serial = 'woaibeijingtiananmenTianmenshangtaiyangsheng'

        # 创建文档模板
        self.doc = BaseDocTemplate(
            output_path,
            pagesize=page_size,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        content_frame = Frame(
            self.doc.leftMargin,
            self.doc.bottomMargin + 0.5 * inch,  # Leave space for footer
            self.doc.width,
            self.doc.height - 0.5 * inch,
            id='content'
        )
        footer_frame = Frame(
            self.doc.leftMargin,
            self.doc.bottomMargin,
            self.doc.width,
            0.5 * inch,
            id='footer'
        )
        self.doc.addPageTemplates([
            PageTemplate(id='FirstPage', frames=[
                         content_frame, footer_frame], onPage=self._render_page),
            PageTemplate(id='LaterPages', frames=[
                         content_frame, footer_frame], onPage=self._later_pages)
        ])

    def _register_chinese_font(self):
        """注册中文字体，返回字体名称"""
        try:
            # 尝试使用系统字体
            font_paths = [
                # Windows
                "C:/Windows/Fonts/msyh.ttc",  # 宋体
                "C:/Windows/Fonts/simsun.ttc",  # 宋体
                "C:/Windows/Fonts/simhei.ttf",  # 黑体
                # MacOS
                "/System/Library/Fonts/STSong.ttf",
                "/System/Library/Fonts/STHeiti Medium.ttc",
                # Linux
                "/usr/share/fonts/wenquanyi/wqy-zenhei.ttc",
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
            ]

            for path in font_paths:
                if os.path.exists(path):
                    font_name = os.path.splitext(os.path.basename(path))[0]
                    try:
                        pdfmetrics.registerFont(TTFont(font_name, path))
                        return font_name
                    except:
                        continue

            # 尝试使用reportlab亚洲字体
            from reportlab.pdfbase.cidfonts import UnicodeCIDFont
            pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
            return 'STSong-Light'
        except:
            raise Exception("无法注册中文字体，请确保系统安装了中文字体")

    def _setup_styles(self):
        """设置样式表"""
        styles = getSampleStyleSheet()

        # 修改现有样式使用中文字体
        for style_name in styles.byName:
            styles[style_name].fontName = self.chinese_font

        # 添加自定义样式
        custom_styles = {
            'Title': {
                'parent': styles['Heading1'],
                'fontSize': 24,
                'leading': 28,
                'alignment': TA_CENTER,
                'spaceAfter': 20,
                'textColor': colors.darkblue,
                'fontName': self.chinese_font
            },
            'Subtitle': {
                'parent': styles['Heading2'],
                'fontSize': 14,
                'leading': 18,
                'alignment': TA_CENTER,
                'spaceAfter': 15,
                'textColor': colors.darkblue,
                'fontName': self.chinese_font
            },
            'BodyText': {
                'parent': styles['BodyText'],
                'fontSize': 12,
                'leading': 15,
                'alignment': TA_LEFT,
                'spaceAfter': 10,
                'fontName': self.chinese_font
            },
            'CenteredText': {
                'parent': styles['Normal'],
                'fontSize': 12,
                'leading': 15,
                'alignment': TA_CENTER,
                'spaceAfter': 10,
                'fontName': self.chinese_font
            },
            'ImageCaption': {
                'parent': styles['Italic'],
                'fontSize': 10,
                'leading': 12,
                'alignment': TA_CENTER,
                'spaceBefore': 5,
                'spaceAfter': 15,
                'fontName': self.chinese_font
            },
            'Stopper': {
                'parent': styles['Normal'],
                'fontSize': 8,
                'leading': 10,
                'alignment': TA_CENTER,
                'spaceBefore': 10,
                'fontName': self.chinese_font
            }
        }

        for style_name, style_params in custom_styles.items():
            if style_name not in styles:
                styles.add(ParagraphStyle(name=style_name, **style_params))

        return styles

    def add_title_page(self, subtitle=None, author=None):
        """添加标题页"""
        # 添加主标题
        self.elements.append(Paragraph(self.title, self.styles['Title']))

        # 添加副标题（如果有）
        if subtitle:
            self.elements.append(Paragraph(subtitle, self.styles['Subtitle']))

        # 添加作者和日期（如果有）
        self.elements.append(Spacer(1, 2*inch))
        if author:
            self.elements.append(
                Paragraph(f"作者: {author}", self.styles['CenteredText']))

        self.elements.append(
            Paragraph(f'序列号：{self.serial}', self.styles['CenteredText']))

        date = datetime.now().strftime("%Y年%m月%d日")
        date = datetime.now().isoformat()
        self.elements.append(
            Paragraph(f'日期：{date}', self.styles['CenteredText']))

        # 添加分页符
        self.elements.append(PageBreak())

    def add_paragraph(self, text, style='BodyText'):
        """添加段落"""
        self.elements.append(Paragraph(text, self.styles[style]))
        self.elements.append(Spacer(1, 0.2*inch))

    def add_image(self, image_path_or_fig, caption=None, width=None, height=None):
        """添加图像

        参数:
            image_path_or_fig: 图像文件路径或matplotlib图形对象
            caption: 图像标题 (可选)
            width: 图像宽度 (可选)
            height: 图像高度 (可选)
        """
        if isinstance(image_path_or_fig, str):
            # 如果是文件路径，加载图像
            with PILImage.open(image_path_or_fig) as img:
                img_width, img_height = img.size
        else:
            # 如果是matplotlib图形对象，保存为字节流
            buf = BytesIO()
            image_path_or_fig.savefig(buf, format='png')
            buf.seek(0)
            with PILImage.open(buf) as img:
                img_width, img_height = img.size

        # 计算保持纵横比的尺寸
        if width and height:
            # 如果同时指定了宽高，使用指定值
            pass
        elif width:
            # 如果只指定了宽度，计算高度保持比例
            height = (width / img_width) * img_height
        elif height:
            # 如果只指定了高度，计算宽度保持比例
            width = (height / img_height) * img_width
        else:
            # 默认宽度为页面宽度的80%
            width = min(self.page_size[0] * 0.8, img_width)
            height = (width / img_width) * img_height

        # 确保图像不会太大
        max_height = self.page_size[1] * 0.6
        if height > max_height:
            height = max_height
            width = (height / img_height) * img_width

        # 添加图像
        if isinstance(image_path_or_fig, str):
            img = Image(image_path_or_fig, width=width, height=height)
        else:
            buf.seek(0)
            img = Image(buf, width=width, height=height)

        self.elements.append(img)

        # 添加图像标题（如果有）
        if caption:
            self.elements.append(
                Paragraph(caption, self.styles['ImageCaption']))

        # 添加一些间距
        self.elements.append(Spacer(1, 0.3*inch))

    def add_page_break(self):
        """添加分页符"""
        self.elements.append(PageBreak())

    def add_stopper(self, text="自动生成报告"):
        """添加页脚"""
        # Ensure the footer fits within the current page
        # Add a small spacer for separation
        self.elements.append(Spacer(1, 0.5 * inch))
        self.elements.append(Paragraph(text, self.styles['Stopper']))

    def _render_page(self, canvas: canvas.Canvas, doc):
        """Customizes the first page (adds footer at the bottom)."""
        canvas.saveState()
        # Footer text
        footer_text = f'序列号：{self.serial.upper()}'  # "Python自动报告生成 - 仅供学习使用"
        canvas.setFont(self.chinese_font, 8)
        canvas.setFillColor('gray')
        canvas.drawCentredString(
            self.page_size[0] / 2.0,
            0.5 * inch,  # Ensure footer is at the bottom
            footer_text
        )
        # Page number
        current_page = doc.page
        page_number = f"第 {current_page} 页"
        canvas.drawCentredString(
            self.page_size[0] / 2.0,
            0.35 * inch,  # Page number below footer text
            page_number
        )
        canvas.restoreState()

    def _later_pages(self, canvas, doc):
        """Adds footer and page numbers to subsequent pages."""
        canvas.saveState()
        # Footer text
        footer_text = "Python自动报告生成 - 仅供学习使用"
        canvas.setFont(self.chinese_font, 8)
        canvas.drawCentredString(
            self.page_size[0] / 2.0,
            0.5 * inch,  # Ensure footer is at the bottom
            footer_text
        )
        # Page number
        current_page = doc.page
        page_number = f"第 {current_page} 页"
        canvas.drawCentredString(
            self.page_size[0] / 2.0,
            0.35 * inch,  # Page number below footer text
            page_number
        )
        canvas.restoreState()

    def generate(self):
        """生成PDF文件"""
        self.doc.build(self.elements)
        print(f"报告已生成: {os.path.abspath(self.output_path)}")

# %% ---- 2025-04-01 ------------------------
# Play ground


# %% ---- 2025-04-01 ------------------------
# Pending


# %% ---- 2025-04-01 ------------------------
# Pending
