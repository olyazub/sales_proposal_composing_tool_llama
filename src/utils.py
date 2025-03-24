import torch
from transformers import pipeline
from IPython.display import clear_output, display, FileLink
import ipywidgets as widgets

def generate_proposal_section(pipe, prompt, max_length=500):
    """Generate content for a proposal section using the Llama model."""
    messages = [
        {"role": "system", "content": "You are an assistant that helps salespeople create professional sales proposals."},
        {"role": "user", "content": prompt}
    ]
    
    # Convert messages to Llama chat format
    formatted_prompt = ""
    for message in messages:
        if message["role"] == "system":
            formatted_prompt += f"<|system|>\n{message['content']}</s>\n"
        elif message["role"] == "user":
            formatted_prompt += f"<|user|>\n{message['content']}</s>\n"
        elif message["role"] == "assistant":
            formatted_prompt += f"<|assistant|>\n{message['content']}</s>\n"
    
    formatted_prompt += "<|assistant|>\n"
    
    # Generate the response
    sequences = pipe(
        formatted_prompt,
        do_sample=True,
        max_length=len(pipe.tokenizer.encode(formatted_prompt)) + max_length,
        temperature=0.7,
        top_p=0.9,
        return_full_text=False
    )
    
    return sequences[0]['generated_text']

def create_section_heading(title):
    """Create an HTML heading for a section."""
    return widgets.HTML(f"<h2>{title}</h2>")

def create_text_area(description, placeholder, width='80%', height='100px'):
    """Create a text area widget with the given properties."""
    return widgets.Textarea(
        description=description,
        placeholder=placeholder,
        layout=widgets.Layout(width=width, height=height)
    )

def create_button(description, button_style=None):
    """Create a button widget with the given properties."""
    if button_style:
        return widgets.Button(description=description, button_style=button_style)
    else:
        return widgets.Button(description=description)

def generate_html_proposal(proposal_data):
    """Generate HTML for the final proposal."""
    html_proposal = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{proposal_data['title']}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }}
            .date {{
                text-align: right;
                color: #7f8c8d;
            }}
            .section {{
                margin-bottom: 25px;
            }}
            .footer {{
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #bdc3c7;
                text-align: center;
                font-size: 0.9em;
                color: #7f8c8d;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{proposal_data['title']}</h1>
            <p>Prepared for: <strong>{proposal_data['customer']}</strong></p>
            <p>Prepared by: {proposal_data['salesperson']}</p>
            <p class="date">{proposal_data['date']}</p>
        </div>
        
        <div class="section">
            <h2>Executive Summary</h2>
            <p>{proposal_data['executive_summary']}</p>
        </div>
        
        <div class="section">
            <h2>Problem Statement</h2>
            <p>{proposal_data['problem_statement']}</p>
        </div>
        
        <div class="section">
            <h2>Proposed Solution</h2>
            <p>{proposal_data['proposed_solution']}</p>
        </div>
        
        <div class="section">
            <h2>Benefits</h2>
            <p>{proposal_data['benefits']}</p>
        </div>
        
        <div class="section">
            <h2>Pricing and Terms</h2>
            <p>{proposal_data['pricing_terms']}</p>
        </div>
        
        <div class="section">
            <h2>Conclusion</h2>
            <p>{proposal_data['conclusion']}</p>
        </div>
        
        <div class="footer">
            <p>© {proposal_data['date'].split(',')[-1].strip() if ',' in proposal_data['date'] else proposal_data['date']} - Confidential</p>
        </div>
    </body>
    </html>
    """
    
    return html_proposal

def save_and_download_html(html_content):
    """Save HTML content to a file and create a download link."""
    with open('sales_proposal.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return FileLink('sales_proposal.html', result_html_prefix="Click here to download: ") 