import pymupdf4llm
import os

def main():
    pdf_filename = "joke-structure-guide.pdf"
    output_filename = "parsed_document.md"
    
    # Verify the file actually exists before trying to parse
    if not os.path.exists(pdf_filename):
        print(f"Error: Could not find '{pdf_filename}' in the current folder.")
        return

    print(f"Parsing '{pdf_filename}' into Markdown...")
    
    try:
        # The magic happens here: this single function handles columns, 
        # extracts tables as Markdown, and preserves formatting.
        md_text = pymupdf4llm.to_markdown(pdf_filename)
        
        # Save the extracted text to a new file so we can inspect it
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(md_text)
            
        print(f"Success! Document parsed.")
        print(f"Saved {len(md_text)} characters to '{output_filename}'.")
        print("\nOpen 'parsed_document.md' in your code editor to verify the formatting stayed intact!")
        
    except Exception as e:
        print(f"❌ An error occurred during parsing: {e}")

if __name__ == "__main__":
    main()