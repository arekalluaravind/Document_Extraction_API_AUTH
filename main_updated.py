import os
import glob
import pandas as pd
from graph_workflow_updated import create_document_router_graph


def save_result_to_excel(extracted_data, output_dir: str, base_filename: str):
    """Flattens and saves Pydantic model data into an Excel spreadsheet."""
    data_dict = extracted_data.model_dump()

    # Create Excel writer
    excel_path = os.path.join(output_dir, f"{base_filename}_extracted.xlsx")

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        # 1. Handle Line Items if present (Invoices & POs)
        if "line_items" in data_dict and data_dict["line_items"]:
            df_items = pd.DataFrame(data_dict["line_items"])
            df_items.to_excel(writer, sheet_name="Line Items", index=False)

        # 2. Handle Summary / Header fields (Excluding nested lists for clean tabular view)
        header_data = {k: v for k, v in data_dict.items() if k != "line_items" and k != "referenced_sections"}
        df_header = pd.DataFrame([header_data])
        df_header.to_excel(writer, sheet_name="Summary & Headers", index=False)

        # 3. Handle Contracts sections if present
        if "referenced_sections" in data_dict and data_dict["referenced_sections"]:
            df_sections = pd.DataFrame({"Referenced Sections": data_dict["referenced_sections"]})
            df_sections.to_excel(writer, sheet_name="Referenced Sections", index=False)

    print(f"📊 Excel report successfully saved at: {excel_path}")


def process_folder(input_folder: str, output_root: str, user_query: str = None):
    os.makedirs(output_root, exist_ok=True)
    pdf_files = glob.glob(os.path.join(input_folder, "*.pdf"))

    if not pdf_files:
        print(f"⚠️ No PDF files found in directory: {input_folder}")
        return

    app = create_document_router_graph()

    for file_path in pdf_files:
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        # Create a dedicated output folder for each file name
        file_output_dir = os.path.join(output_root, base_name)
        os.makedirs(file_output_dir, exist_ok=True)

        print(f"\n==========================================")
        print(f"📂 Processing File: {file_path}")
        print(f"==========================================")

        initial_state = {
            "pdf_path": file_path,
            "user_query": user_query,
            "extracted_text": "",
            "extraction_method": "",
            "uploaded_file": None,
            "document_type": None,
            "extracted_data": None
        }

        try:
            result = app.invoke(initial_state)
            extracted_data = result["extracted_data"]

            print(f"\n✨ **Final Structured Output Result for {base_name}:**")
            print(extracted_data.model_dump_json(indent=2))

            # Save output as Excel inside the file's folder
            save_result_to_excel(extracted_data, file_output_dir, base_name)

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")


if __name__ == "__main__":
    INPUT_FOLDER = "./input_documents"  # Place your input PDFs here
    OUTPUT_FOLDER = "./output_results"  # Folder where subfolders & Excel files will be created

    # Example run with batch folder processing
    process_folder(
        input_folder=INPUT_FOLDER,
        output_root=OUTPUT_FOLDER,
        user_query= None
    )