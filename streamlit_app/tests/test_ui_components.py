import unittest
from unittest.mock import patch, MagicMock
from PIL import Image

# Assuming 'ui_components' is correctly imported and accessible
import ui_components

class TestUIComponents(unittest.TestCase):

    @patch('ui_components.st')
    @patch('ui_components.sqlite3.connect')

    def test_feedback_form(self, mock_connect, mock_st):
        # Setup mocks for the database and form handling
        mock_conn = mock_connect.return_value.__enter__.return_value
        mock_cursor = mock_conn.cursor.return_value
        mock_st.form.return_value.__enter__.return_value.submit_feedback = True
        mock_st.radio.return_value = "Yes"
        mock_st.slider.return_value = 5
        mock_st.text_area.return_value = "Great job!"

        # Call the function
        ui_components.feedback_form()

        # Assertions to verify behavior
        mock_connect.assert_called_once_with('feedback.db')
        mock_cursor.execute.assert_called_once()  # Add specific SQL query check if possible
        mock_conn.commit.assert_called_once()
        mock_st.success.assert_called_once_with('Thank you for your feedback!')
    
    @patch('ui_components.st')
    def test_setup_page(self, mock_st):
        ui_components.setup_page()
        mock_st.set_page_config.assert_called_once_with(page_title="PureView - Image Processing App", page_icon=":camera:", layout="wide")
        mock_st.markdown.assert_called_once()
        mock_st.title.assert_called_once_with("Welcome to PureView!")

    @patch('ui_components.st')
    @patch('ui_components.Image.open')
    @patch('ui_components.classify_image')
    @patch('ui_components.deblur_and_enhance_image')
    def test_upload_and_display_image(self, mock_deblur, mock_classify, mock_open, mock_st):
        mock_st.file_uploader.return_value = 'fake_path'
        mock_open.return_value = MagicMock(spec=Image.Image)
        mock_classify.return_value = 'blur_type'
        mock_deblur.return_value = MagicMock(spec=Image.Image)
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_st.columns.return_value = (mock_col1, mock_col2)

        original_image, blur_type = ui_components.upload_and_display_image()

        mock_st.file_uploader.assert_called_once_with("Choose an image...", type=["jpg", "jpeg", "png"])
        mock_open.assert_called_once_with('fake_path')
        mock_classify.assert_called_once()
        mock_deblur.assert_called_once()
        mock_st.columns.assert_called_once_with(2, gap="medium")
        mock_col1.image.assert_called()
        mock_col2.image.assert_called()

if __name__ == '__main__':
    unittest.main()
