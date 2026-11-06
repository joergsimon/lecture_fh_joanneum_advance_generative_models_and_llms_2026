from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "handouts" / "Autoregression_Blackboard_Cheat_Sheet.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

GREEN = colors.HexColor("#78BE20")
MAGENTA = colors.HexColor("#D71970")
INK = colors.HexColor("#17212B")
MUTED = colors.HexColor("#55616D")
PALE = colors.HexColor("#F3F6F7")
LINE = colors.HexColor("#CBD3D7")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="TitleFH", parent=styles["Title"], fontName="Helvetica-Bold",
    fontSize=22, leading=25, textColor=INK, alignment=TA_LEFT, spaceAfter=3 * mm,
))
styles.add(ParagraphStyle(
    name="Deck", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=9.5, leading=13, textColor=MUTED, spaceAfter=4 * mm,
))
styles.add(ParagraphStyle(
    name="H2FH", parent=styles["Heading2"], fontName="Helvetica-Bold",
    fontSize=12.5, leading=15, textColor=INK, spaceBefore=2 * mm, spaceAfter=2 * mm,
))
styles.add(ParagraphStyle(
    name="BodyFH", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=9.2, leading=12.2, textColor=INK, spaceAfter=1.5 * mm,
))
styles.add(ParagraphStyle(
    name="SmallFH", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=8.3, leading=10.5, textColor=MUTED,
))
styles.add(ParagraphStyle(
    name="TableHeadFH", parent=styles["BodyText"], fontName="Helvetica-Bold",
    fontSize=8.3, leading=10.5, textColor=colors.white,
))
styles.add(ParagraphStyle(
    name="FormulaFH", parent=styles["BodyText"], fontName="Helvetica-Bold",
    fontSize=13, leading=17, textColor=INK, alignment=TA_CENTER,
))


def P(text, style="BodyFH"):
    return Paragraph(text, styles[style])


doc = SimpleDocTemplate(
    str(OUT), pagesize=A4,
    rightMargin=15 * mm, leftMargin=15 * mm,
    topMargin=13 * mm, bottomMargin=13 * mm,
    title="Autoregression: four-pixel blackboard example",
    author="Advanced Generative Methods and Large Language Models",
)

story = []
story.append(Table([["", "LECTURE 2 · INSTRUCTOR SHEET"]], colWidths=[5 * mm, 170 * mm],
                   style=TableStyle([
                       ("BACKGROUND", (0, 0), (0, 0), GREEN),
                       ("TEXTCOLOR", (1, 0), (1, 0), MUTED),
                       ("FONTNAME", (1, 0), (1, 0), "Helvetica-Bold"),
                       ("FONTSIZE", (1, 0), (1, 0), 8),
                       ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                       ("LEFTPADDING", (1, 0), (1, 0), 3 * mm),
                   ])))
story.append(Spacer(1, 4 * mm))
story.append(P("Autoregression on four pixels", "TitleFH"))
story.append(P(
    "A compact worked example for the blackboard. The point is the chain rule: a fixed image has an "
    "exactly evaluable probability, while generating an image remains stochastic and sequential.", "Deck"))

story.append(P("1 · Draw the observed image and the scan order", "H2FH"))
pixel_table = Table(
    [[P("<b>x1 = 1</b>", "FormulaFH"), P("<b>x2 = 0</b>", "FormulaFH"),
      P("<b>x3 = 1</b>", "FormulaFH"), P("<b>x4 = 1</b>", "FormulaFH")],
     [P("first", "SmallFH"), P("condition on x1", "SmallFH"),
      P("condition on x1,x2", "SmallFH"), P("condition on x1,x2,x3", "SmallFH")]],
    colWidths=[45 * mm] * 4,
)
pixel_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (0, 0), GREEN),
    ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#E8EDF0")),
    ("BACKGROUND", (2, 0), (3, 0), GREEN),
    ("GRID", (0, 0), (-1, 0), 0.7, colors.white),
    ("BOX", (0, 0), (-1, -1), 0.7, LINE),
    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, 0), 4 * mm),
    ("BOTTOMPADDING", (0, 0), (-1, 0), 4 * mm),
    ("TOPPADDING", (0, 1), (-1, 1), 2 * mm),
    ("BOTTOMPADDING", (0, 1), (-1, 1), 2 * mm),
]))
story.append(pixel_table)
story.append(Spacer(1, 3 * mm))
story.append(Table([[P(
    "p(x1,x2,x3,x4) = p(x1) · p(x2 | x1) · p(x3 | x1,x2) · p(x4 | x1,x2,x3)",
    "FormulaFH")]], colWidths=[180 * mm], style=TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE),
        ("BOX", (0, 0), (-1, -1), 0.8, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 3 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3 * mm),
    ])))

story.append(P("2 · Use the model's four conditional probabilities", "H2FH"))
rows = [
    [P("Observed value", "TableHeadFH"), P("Model output used", "TableHeadFH"), P("Factor", "TableHeadFH")],
    [P("x1 = 1"), P("p(x1=1) = 0.80"), P("0.80")],
    [P("x2 = 0"), P("p(x2=0 | x1=1) = 1 - p(x2=1 | x1=1)<br/>= 1 - 0.25"), P("0.75")],
    [P("x3 = 1"), P("p(x3=1 | x1=1,x2=0) = 0.70"), P("0.70")],
    [P("x4 = 1"), P("p(x4=1 | x1=1,x2=0,x3=1) = 0.90"), P("0.90")],
]
t = Table(rows, colWidths=[35 * mm, 110 * mm, 35 * mm], repeatRows=1)
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), INK),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.5, LINE),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("LEFTPADDING", (0, 0), (-1, -1), 2.5 * mm),
    ("RIGHTPADDING", (0, 0), (-1, -1), 2.5 * mm),
    ("TOPPADDING", (0, 0), (-1, -1), 1.8 * mm),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8 * mm),
    ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#FFF4F8")),
]))
story.append(t)

story.append(P("3 · Multiply; then take the negative log", "H2FH"))
calc = Table([
    [P("Image likelihood", "SmallFH"), P("0.80 × 0.75 × 0.70 × 0.90 = <b>0.378</b>", "FormulaFH")],
    [P("Negative log-likelihood", "SmallFH"), P("-ln(0.378) = <b>0.9729 ≈ 0.973 nats</b>", "FormulaFH")],
], colWidths=[48 * mm, 132 * mm])
calc.setStyle(TableStyle([
    ("BOX", (0, 0), (-1, -1), 0.8, LINE),
    ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE),
    ("BACKGROUND", (0, 0), (0, -1), PALE),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 3 * mm),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3 * mm),
]))
story.append(calc)
story.append(Spacer(1, 2 * mm))

notes = Table([
    [P("<b>Exact likelihood</b>", "BodyFH"), P(
        "For this fixed complete image, all normalized factors can be evaluated and multiplied. "
        "No partition function, variational bound, or Monte Carlo estimate is needed.")],
    [P("<b>Not deterministic generation</b>", "BodyFH"), P(
        "Sampling still draws x1, then x2 conditioned on the sampled x1, and so on. A different random seed can yield a different image. Greedy decoding would be a separate choice.")],
], colWidths=[45 * mm, 135 * mm])
notes.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFA")),
    ("BOX", (0, 0), (-1, -1), 0.8, LINE),
    ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("LEFTPADDING", (0, 0), (-1, -1), 3 * mm),
    ("RIGHTPADDING", (0, 0), (-1, -1), 3 * mm),
    ("TOPPADDING", (0, 0), (-1, -1), 2.5 * mm),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5 * mm),
]))
story.append(notes)

story.append(P("Blackboard prompts", "H2FH"))
story.append(P(
    "1. Ask why the second factor is 0.75 rather than 0.25. &nbsp; "
    "2. Ask which later conditionals may change if x2 changes. &nbsp; "
    "3. Contrast parallel likelihood evaluation during training with sequential sampling at inference. &nbsp; "
    "4. End with: exact probability of a fixed image does not mean the generated image is fixed.", "SmallFH"))

doc.build(story)
print(OUT)
