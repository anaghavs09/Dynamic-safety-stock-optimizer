import markdown
import os

def render_markdown_to_html(md_file='README.md', html_file='README.html'):
    if not os.path.exists(md_file):
        print(f"Error: {md_file} not found.")
        return

    with open(md_file, 'r', encoding='utf-8') as f:
        text = f.read()

    html_body = markdown.markdown(text, extensions=['fenced_code', 'tables'])

    full_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Inventory Optimizer - Technical Documentation</title>
        <style>
            :root {{
                --bg: #ffffff;
                --text: #1a1a1a;
                --text-muted: #57606a;
                --border: #d0d7de;
                --code-bg: #f6f8fa;
                --link: #0969da;
            }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                line-height: 1.625;
                color: var(--text);
                max-width: 800px;
                margin: 60px auto;
                padding: 0 40px;
                background-color: var(--bg);
                -webkit-font-smoothing: antialiased;
            }}
            h1, h2, h3 {{
                margin-top: 2rem;
                margin-bottom: 1rem;
                font-weight: 600;
                line-height: 1.25;
                color: #000;
            }}
            h1 {{
                font-size: 2.25rem;
                border-bottom: 1px solid var(--border);
                padding-bottom: 0.5rem;
            }}
            h2 {{
                font-size: 1.5rem;
                border-bottom: 1px solid var(--border);
                padding-bottom: 0.3rem;
            }}
            p, ul, ol {{
                margin-top: 0;
                margin-bottom: 1.25rem;
            }}
            code {{
                padding: 0.2rem 0.4rem;
                font-size: 85%;
                background-color: var(--code-bg);
                border-radius: 6px;
                font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
            }}
            pre {{
                padding: 1.25rem;
                overflow: auto;
                font-size: 90%;
                line-height: 1.45;
                background-color: var(--code-bg);
                border-radius: 8px;
                margin-bottom: 1.5rem;
                border: 1px solid var(--border);
            }}
            pre code {{
                background-color: transparent;
                padding: 0;
                border-radius: 0;
            }}
            ul {{
                padding-left: 1.5rem;
            }}
            li {{
                margin-bottom: 0.5rem;
            }}
            hr {{
                height: 0.25rem;
                padding: 0;
                margin: 3rem 0;
                background-color: var(--border);
                border: 0;
            }}
            blockquote {{
                padding: 0 1rem;
                color: var(--text-muted);
                border-left: 0.25rem solid var(--border);
                margin: 0 0 1.25rem 0;
            }}
        </style>
    </head>
    <body>
        {html_body}
    </body>
    </html>
    """

    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(full_html)
    print(f"✅ README rendered to {html_file}")

if __name__ == "__main__":
    render_markdown_to_html()
