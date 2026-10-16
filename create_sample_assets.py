import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from PIL import Image, ImageDraw, ImageFont


def create_sample_pdf(output_path: str):
    """Generate a 4-page academic study material PDF on Machine Learning."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=12
    )
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#2563EB'),
        spaceBefore=10,
        spaceAfter=8
    )
    body_style = ParagraphStyle(
        'BodyText',
        parent=styles['Normal'],
        fontSize=10,
        leading=15,
        textColor=colors.HexColor('#334155'),
        spaceAfter=10
    )

    story = []

    # PAGE 1: Machine Learning Fundamentals
    story.append(Paragraph("Chapter 1: Machine Learning Fundamentals", title_style))
    story.append(Paragraph("1.1 Introduction to Learning Paradigms", heading_style))
    story.append(Paragraph(
        "Machine Learning is a subset of artificial intelligence where algorithms improve their performance "
        "on tasks through statistical inference and data experience rather than explicit deterministic programming. "
        "The two fundamental paradigms are Supervised Learning and Unsupervised Learning.",
        body_style
    ))
    story.append(Paragraph("1.2 Supervised vs Unsupervised Learning", heading_style))
    story.append(Paragraph(
        "In <b>Supervised Learning</b>, the algorithm learns from a labeled dataset where each training input is paired "
        "with an explicit ground-truth target. Common tasks include classification (predicting discrete classes, such as "
        "spam detection or medical diagnosis) and regression (predicting continuous numerical quantities, such as house prices). "
        "Popular supervised algorithms include Linear Regression, Support Vector Machines (SVM), and Random Forests.",
        body_style
    ))
    story.append(Paragraph(
        "In contrast, <b>Unsupervised Learning</b> deals with unlabeled data where the model must autonomously discover "
        "hidden patterns, groupings, or geometric structures without prior human guidance. Key applications include "
        "clustering (e.g., K-Means, DBSCAN for customer segmentation), dimensionality reduction (e.g., Principal Component "
        "Analysis - PCA), and anomaly detection in cybersecurity.",
        body_style
    ))
    story.append(PageBreak())

    # PAGE 2: Neural Networks & Backpropagation
    story.append(Paragraph("Chapter 2: Deep Learning & Neural Networks", title_style))
    story.append(Paragraph("2.1 Multilayer Perceptrons and Activation Functions", heading_style))
    story.append(Paragraph(
        "Artificial Neural Networks (ANNs) are computational models inspired by biological neural networks. "
        "A Multilayer Perceptron (MLP) consists of an input layer, one or more hidden layers, and an output layer. "
        "Non-linear activation functions such as ReLU (Rectified Linear Unit), Sigmoid, and GELU allow the network "
        "to approximate arbitrary non-linear mathematical mappings.",
        body_style
    ))
    story.append(Paragraph("2.2 The Backpropagation Algorithm", heading_style))
    story.append(Paragraph(
        "Backpropagation is the cornerstone optimization algorithm for training deep neural networks. It systematically "
        "computes the gradient of the scalar loss function with respect to every weight parameter in the network using the "
        "multivariate chain rule of calculus. These gradients are subsequently used by optimization algorithms such as "
        "Stochastic Gradient Descent (SGD) or Adam to iteratively update the network parameters and minimize prediction error.",
        body_style
    ))
    story.append(PageBreak())

    # PAGE 3: Convolutional Neural Networks
    story.append(Paragraph("Chapter 3: Convolutional Neural Networks (CNN)", title_style))
    story.append(Paragraph("3.1 Architecture of CNNs", heading_style))
    story.append(Paragraph(
        "Convolutional Neural Networks are specialized neural architectures designed specifically for processing grid-structured "
        "data such as 2D digital images. The primary building blocks of CNNs include convolutional layers, non-linear activation "
        "layers, pooling layers, and fully connected dense layers.",
        body_style
    ))
    story.append(Paragraph("3.2 Function and Significance of Pooling Layers", heading_style))
    story.append(Paragraph(
        "The primary purpose of a <b>pooling layer</b> is to reduce the spatial dimensions (height and width) of the feature maps, "
        "thereby drastically reducing the number of computational parameters and mathematical operations required in subsequent layers. "
        "This downsampling mechanism achieves spatial translational invariance, making the detected visual features robust against "
        "minor shifts, rotations, and distortions in the input image. The most prevalent technique is <b>Max Pooling</b>, which "
        "slides a rectangular filter window across the input feature map and selects the maximum activation value within each window.",
        body_style
    ))
    story.append(PageBreak())

    # PAGE 4: Natural Language Processing & Transformers
    story.append(Paragraph("Chapter 4: Attention Mechanisms & Transformers", title_style))
    story.append(Paragraph("4.1 Self-Attention Mechanism", heading_style))
    story.append(Paragraph(
        "The Transformer architecture, introduced by Vaswani et al. in 2017, completely revolutionized modern Natural Language "
        "Processing by replacing recurrence with self-attention. The self-attention mechanism enables the model to weigh the "
        "semantic importance of all other tokens in a sequence simultaneously when computing the contextual representation of a target token.",
        body_style
    ))
    story.append(Paragraph("4.2 Query, Key, and Value Vectors", heading_style))
    story.append(Paragraph(
        "In scaled dot-product attention, each input embedding is linearly projected into three distinct vectors: Query (Q), "
        "Key (K), and Value (V). The attention score between query and key vectors is computed using the dot product, scaled by the "
        "square root of the key dimension (sqrt(d_k)), normalized through a Softmax function, and then multiplied by the Value vectors "
        "to form the synthesized contextual output.",
        body_style
    ))

    doc.build(story)
    print(f"Sample PDF created at: {output_path}")


def create_sample_question_image(output_path: str, question_text: str):
    """Create a high-contrast, clean image containing a sample academic question for OCR testing."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width = 900
    height = 250
    # Create white canvas
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw border
    draw.rectangle([10, 10, width - 10, height - 10], outline=(200, 210, 225), width=3)

    # Draw Title
    draw.text((40, 30), "EXAM QUESTION / LAB ASSIGNMENT", fill=(100, 116, 139))

    # Split question into lines if long
    words = question_text.split()
    lines = []
    current_line = []
    for w in words:
        current_line.append(w)
        if len(" ".join(current_line)) > 45:
            lines.append(" ".join(current_line))
            current_line = []
    if current_line:
        lines.append(" ".join(current_line))

    y_offset = 80
    for line in lines:
        draw.text((40, y_offset), line, fill=(15, 23, 42))
        y_offset += 40

    img.save(output_path)
    print(f"Sample question image created at: {output_path}")


if __name__ == "__main__":
    pdf_dest = os.path.join("sample_documents", "machine_learning_notes.pdf")
    q1_dest = os.path.join("sample_documents", "question_pooling_layers.png")
    q2_dest = os.path.join("sample_documents", "question_supervised_vs_unsupervised.png")

    create_sample_pdf(pdf_dest)
    create_sample_question_image(
        q1_dest,
        "What is the function of pooling layers in Convolutional Neural Networks?"
    )
    create_sample_question_image(
        q2_dest,
        "What is the difference between supervised and unsupervised learning?"
    )
