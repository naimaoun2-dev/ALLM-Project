"""
Download Handler for PDF and Excel Export
"""
import pandas as pd
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT
from datetime import datetime

class DownloadHandler:
    """Handle PDF and Excel downloads of chat history"""
    
    def generate_pdf(self, chat_history: list) -> bytes:
        """
        Generate PDF from chat history
        
        Args:
            chat_history: List of chat messages
            
        Returns:
            PDF file as bytes
        """
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = []
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor='#1f77b4',
            spaceAfter=30,
            alignment=TA_LEFT
        )
        
        user_style = ParagraphStyle(
            'UserStyle',
            parent=styles['Normal'],
            fontSize=11,
            textColor='#2c3e50',
            leftIndent=20,
            spaceAfter=10,
            backColor='#ecf0f1',
            borderPadding=10
        )
        
        assistant_style = ParagraphStyle(
            'AssistantStyle',
            parent=styles['Normal'],
            fontSize=11,
            textColor='#27ae60',
            leftIndent=20,
            spaceAfter=10,
            backColor='#d5f4e6',
            borderPadding=10
        )
        
        # Title
        story.append(Paragraph("Chat History", title_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Chat messages
        for i, message in enumerate(chat_history):
            role = message.get("role", "unknown").upper()
            content = message.get("content", "")
            timestamp = message.get("timestamp", "")
            source = message.get("source", "")
            
            # Role header
            role_text = f"<b>{role}</b>"
            if timestamp:
                role_text += f" <i>({timestamp})</i>"
            info_type = message.get("info_type", "")
            if info_type and info_type != "unknown":
                role_text += f" [Info Type: {info_type}]"
            elif source and source != "unknown":
                role_text += f" [Source: {source}]"
            
            story.append(Paragraph(role_text, styles['Heading3']))
            
            # Content
            if role == "USER":
                story.append(Paragraph(content.replace('\n', '<br/>'), user_style))
            else:
                story.append(Paragraph(content.replace('\n', '<br/>'), assistant_style))
            
            story.append(Spacer(1, 0.2*inch))
            
            # Page break every 10 messages
            if (i + 1) % 10 == 0:
                story.append(PageBreak())
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    
    def generate_excel(self, chat_history: list) -> bytes:
        """
        Generate Excel file from chat history
        
        Args:
            chat_history: List of chat messages
            
        Returns:
            Excel file as bytes
        """
        # Prepare data
        data = []
        for message in chat_history:
            data.append({
                "Role": message.get("role", "unknown").upper(),
                "Content": message.get("content", ""),
                "Timestamp": message.get("timestamp", ""),
                "Source": message.get("source", "unknown"),
                "Info Type": message.get("info_type", "unknown")
            })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Create Excel file
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Chat History')
            
            # Get worksheet for formatting
            worksheet = writer.sheets['Chat History']
            
            # Auto-adjust column widths
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 100)
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        buffer.seek(0)
        return buffer.getvalue()

