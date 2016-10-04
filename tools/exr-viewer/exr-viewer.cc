#include <QApplication>
#include <QMainWindow>
#include <QFileDialog>
#include <QMessageBox>
#include <QRadioButton>

#include "ui_exr-viewer.h"
#include "tempest/image/exr-image.hh"
#include "tempest/utils/file-system.hh"
#include "tempest/graphics/texture.hh"
#include "tempest/math/functions.hh"

class EXRViewer: public QMainWindow
{
	Q_OBJECT

	Ui::EXRViewer                                   m_UI;

    QString                                         m_PreviousFile;
    QString                                         m_NextFile;
    uint                                            m_CurrentFile;

public:
	EXRViewer()
	{
		m_UI.setupUi(this);

        m_UI.actionPrevious_Image->setEnabled(false);
        m_UI.actionNext_Image->setEnabled(false);

        m_UI.ScaleSpinBox->setValue(m_UI.ImageView->getScale());
        m_UI.OffsetSpinBox->setValue(m_UI.ImageView->getOffset());
	}

    void openFile(const QString& filename)
    {
        auto status = m_UI.ImageView->open(filename);
        if(!status)
        {
            QMessageBox::critical(this, "Open Failed", QString("Failed to open the specified file: %1").arg(filename));
			return;
        }

        auto cur_dir = QFileInfo(filename).absoluteDir();
        bool enable_navigation = true;
        
        m_CurrentFile = 0;
        auto file = QFileInfo(filename).fileName();
        while(m_CurrentFile < cur_dir.count())
        {
            auto cur_filename = cur_dir[m_CurrentFile];
            if(file == cur_filename)
                break;
            ++m_CurrentFile;
        }
        if(m_CurrentFile == cur_dir.count())
        {
            enable_navigation = false;
        }

        m_PreviousFile.clear();
        m_NextFile.clear();
        uint previous_i;
        for(previous_i = m_CurrentFile - 1; previous_i > 1; --previous_i)
        {
            auto cur_filename = cur_dir[previous_i];
            if(QFileInfo(cur_filename).completeSuffix() == "exr")
            {
                m_PreviousFile = cur_dir.absoluteFilePath(cur_filename);
                break;
            }
        }

        uint next_i;
        for(next_i = m_CurrentFile + 1; next_i < cur_dir.count(); ++next_i)
        {
            auto cur_filename = cur_dir[next_i];
            if(QFileInfo(cur_filename).completeSuffix() == "exr")
            {
                m_NextFile = cur_dir.absoluteFilePath(cur_filename);
                break;
            }
        }

        if(m_PreviousFile.isEmpty() && m_NextFile.isEmpty())
        {
            enable_navigation = false;
        }
        if(m_PreviousFile.isEmpty())
        {
            m_PreviousFile = m_NextFile;
            for(uint i = next_i + 1; i < cur_dir.count(); ++i)
            {
                auto cur_filename = cur_dir[i];
                if(QFileInfo(cur_filename).completeSuffix() == "exr")
                {
                    m_PreviousFile = cur_dir.absoluteFilePath(cur_filename);
                }
            }
        }
        else if(m_NextFile.isEmpty())
        {
            m_NextFile = m_PreviousFile;
            for(uint i = previous_i - 1; i > 1; --i)
            {
                auto cur_filename = cur_dir[i];
                if(QFileInfo(cur_filename).completeSuffix() == "exr")
                {
                    m_NextFile = cur_dir.absoluteFilePath(cur_filename);
                }
            }
        }

        setWindowTitle(QString("Tempest EXR Viewer - %1").arg(file));

        m_UI.actionPrevious_Image->setEnabled(enable_navigation);
        m_UI.actionNext_Image->setEnabled(enable_navigation);

        auto chan_count = m_UI.ImageView->getChannelCount();
        bool enable_alt_view = chan_count == 4;
        m_UI.TangentRadioButton->setEnabled(enable_alt_view);
        m_UI.BinormalRadioButton->setEnabled(enable_alt_view);
        m_UI.NormalRadioButton->setEnabled(enable_alt_view);

        bool enable_single_view = chan_count == 1;
        m_UI.HeatRadioButton->setEnabled(enable_single_view);
        m_UI.LogHeatRadioButton->setEnabled(enable_single_view);

        if((m_UI.HeatRadioButton->isChecked() || m_UI.LogHeatRadioButton->isChecked()) && !enable_single_view ||
           (m_UI.TangentRadioButton->isChecked() || m_UI.BinormalRadioButton->isChecked() ||  m_UI.NormalRadioButton->isChecked()) && !enable_alt_view)
            m_UI.OriginalRadioButton->click();
    }

private slots:
	void on_actionOpen_triggered(bool checked = false)
	{
		QString filename = QFileDialog::getOpenFileName(this, "Open EXR File", {}, "EXR Images (*.exr)");
		if(filename.isNull())
			return;

        openFile(filename);
	}

    void on_actionPrevious_Image_triggered(bool checked = false)
    {
        openFile(m_PreviousFile);
    }

    void on_actionNext_Image_triggered(bool checked = false)
    {
        openFile(m_NextFile);
    }

    void on_actionImage_Manipulation_triggered(bool checked = false)
    {
        m_UI.ImageManipulationDock->setVisible(checked);
    }

    void on_ImageView_hovered(int image_x, int image_y, float* orig_value, float* displayed_value, unsigned component_count)
    {        
        if(image_x == -1 || image_y == -1 || component_count == 0)
        {
            m_UI.statusbar->clearMessage();
            return;
        }

        std::stringstream ss;
        ss << "Image Coordinates = (" << image_x << ", " << image_y << "); Original Value = (";
        for(uint32_t chan_idx = 0; chan_idx < component_count; ++chan_idx)
        {
            ss << orig_value[chan_idx];
            if(chan_idx != component_count - 1)
                ss << ", ";
        }
        ss << "); Displayed Value = (";
        for(uint32_t chan_idx = 0; chan_idx < component_count; ++chan_idx)
        {
            ss << displayed_value[chan_idx];
            if(chan_idx != component_count - 1)
                ss << ", ";
        }
        ss << ")";
        m_UI.statusbar->showMessage(QString::fromStdString(ss.str()));
    }

    void on_ScaleSpinBox_valueChanged(double d)
    {
        m_UI.ImageView->setScale(static_cast<float>(d));
    }

    void on_OffsetSpinBox_valueChanged(double d)
    {
        m_UI.ImageView->setOffset(static_cast<float>(d));
    }

    void on_OriginalRadioButton_toggled(bool checked)
    {
        if(!checked)
            return;

        m_UI.ImageView->setExtractMode(ExtractMode::None);
    }

    void on_TangentRadioButton_toggled(bool checked)
    {
        if(!checked)
            return;

        m_UI.ImageView->setExtractMode(ExtractMode::Tangent);
    }

    void on_BinormalRadioButton_toggled(bool checked)
    {
        if(!checked)
            return;

        m_UI.ImageView->setExtractMode(ExtractMode::Binormal);
    }

    void on_NormalRadioButton_toggled(bool checked)
    {
        if(!checked)
            return;

        m_UI.ImageView->setExtractMode(ExtractMode::Normal);
    }

    void on_LogRadioButton_toggled(bool checked)
    {
        if(!checked)
            return;

        m_UI.ImageView->setExtractMode(ExtractMode::Log);
    }

    void on_HeatRadioButton_toggled(bool checked)
    {
        if(!checked)
            return;

        m_UI.ImageView->setExtractMode(ExtractMode::Heat);
    }

    void on_LogHeatRadioButton_toggled(bool checked)
    {
        if(!checked)
            return;

        m_UI.ImageView->setExtractMode(ExtractMode::LogHeat);
    }

    void on_ManualRadioButton_toggled(bool checked)
    {
        if(!checked)
            return;

        m_UI.ImageView->setNormalization(NormalizationMode::Manual);
        m_UI.ImageView->setScale(m_UI.ScaleSpinBox->value());
        m_UI.ImageView->setOffset(m_UI.OffsetSpinBox->value());
        m_UI.ScaleSpinBox->setEnabled(true);
        m_UI.OffsetSpinBox->setEnabled(true);
    }

    void on_ScaleRangeRadioButton_toggled(bool checked)
    {
        if(!checked)
            return;

        m_UI.ImageView->setNormalization(NormalizationMode::ScaleToRange);
        m_UI.ScaleSpinBox->setEnabled(false);
        m_UI.OffsetSpinBox->setEnabled(false);
    }

    void on_FitRangeRadioButton_toggled(bool checked)
    {
        if(!checked)
            return;

        m_UI.ImageView->setNormalization(NormalizationMode::FitToRange);
        m_UI.ScaleSpinBox->setEnabled(false);
        m_UI.OffsetSpinBox->setEnabled(false);
    }
};

#undef Bool
#include "exr-viewer.moc"

int TempestMain(int argc, char** argv)
{
	QApplication app(argc, argv);

    EXRViewer wnd;

    wnd.show();

    for(int arg = 1; arg < argc; ++arg)
    {
        if(!strcmp(argv[arg] + strlen(argv[arg]) - 4, ".exr"))
        {
            wnd.openFile(argv[arg]);
        }
    }

    return app.exec();
}
