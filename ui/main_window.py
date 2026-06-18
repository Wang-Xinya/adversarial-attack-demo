import sys
import random
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget,
                             QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox,
                             QTextEdit, QProgressBar, QMessageBox, QFileDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from models.lenet import MNISTLeNet
from data.mnist import get_mnist_loader
from utils.common import set_seed, resolve_device, get_single_image, get_prediction
from utils.train import prepare_checkpoint, load_lenet_state_dict
from attacks.single import single_attack
from defenses.eval_batch import evaluate_batch

DATA_DIR = Path("./data")
CHECKPOINT = Path("./lenet_mnist_model.pth")
BATCH_SIZE = 256
DEVICE = "auto"
SEED = 0
NUM_WORKERS = 0
TRAIN_EPOCHS = 3
TRAIN_BATCH_SIZE = 128
FORCE_TRAIN = False
EPSILONS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

class BatchWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object, object, object) 
    error = pyqtSignal(str)

    def __init__(self, model, loader, device, attack_type, defense_params, attack_params):
        super().__init__()
        self.model = model
        self.loader = loader
        self.device = device
        self.attack_type = attack_type
        self.defense_params = defense_params
        self.attack_params = attack_params

    def run(self):
        try:
            if self.defense_params == None:
                fig_ex, fig_curve, fig_diff, _, _ = evaluate_batch(
                    self.model, self.loader, self.device, EPSILONS,
                    attack_type=self.attack_type,
                    defense_type='None',
                    defense_params=self.defense_params,
                    attack_params=self.attack_params
                )
            else:
                fig_ex, fig_curve, fig_diff, _, _ = evaluate_batch(
                    self.model, self.loader, self.device, EPSILONS,
                    attack_type=self.attack_type,
                    defense_type='random_smooth',
                    defense_params=self.defense_params,
                    attack_params=self.attack_params
                )
            self.finished.emit(fig_ex, fig_curve, fig_diff)
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Adversarial Demo System")
        self.setGeometry(100, 100, 1200, 800)

        set_seed(SEED)
        self.device = resolve_device(DEVICE)
        self.prepare_model()
        self.loader = get_mnist_loader(DATA_DIR, BATCH_SIZE, train=False, num_workers=NUM_WORKERS)
        self.total_test_samples = len(self.loader.dataset)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tab_single = QWidget()
        self.tabs.addTab(self.tab_single, "单张图片攻击")
        self.setup_single_tab()

        self.tab_batch = QWidget()
        self.tabs.addTab(self.tab_batch, "批量攻击与防御")
        self.setup_batch_tab()

    def prepare_model(self):
        prepare_checkpoint(
            CHECKPOINT, DATA_DIR, self.device,
            train_epochs=TRAIN_EPOCHS,
            train_batch_size=TRAIN_BATCH_SIZE,
            num_workers=NUM_WORKERS,
            force_train=FORCE_TRAIN
        )
        self.model = MNISTLeNet().to(self.device)
        load_lenet_state_dict(self.model, CHECKPOINT, self.device)
        self.model.eval()

    def setup_single_tab(self):
        layout = QVBoxLayout(self.tab_single)

        control_group = QGroupBox("攻击控制")
        control_layout = QHBoxLayout()

        self.single_attack_type = QComboBox()
        self.single_attack_type.addItems(["FGSM", "PGD", "MI-FGSM"])
        control_layout.addWidget(QLabel("攻击方法:"))
        control_layout.addWidget(self.single_attack_type)

        control_layout.addWidget(QLabel("扰动幅度ε:"))
        self.single_eps = QDoubleSpinBox()
        self.single_eps.setRange(0.0, 1.0)
        self.single_eps.setSingleStep(0.05)
        self.single_eps.setValue(0.1)
        control_layout.addWidget(self.single_eps)

        control_layout.addWidget(QLabel("PGD步数:"))
        self.single_steps = QSpinBox()
        self.single_steps.setRange(1, 50)
        self.single_steps.setValue(10)
        control_layout.addWidget(self.single_steps)

        control_layout.addWidget(QLabel("MI-FGSM衰减因子:"))
        self.single_decay = QDoubleSpinBox()
        self.single_decay.setRange(0.0, 2.0)
        self.single_decay.setSingleStep(0.1)
        self.single_decay.setValue(1.0)
        control_layout.addWidget(self.single_decay)

        self.btn_load_random = QPushButton("随机加载图像")
        self.btn_load_random.clicked.connect(self.load_random_image)
        control_layout.addWidget(self.btn_load_random)

        self.btn_attack = QPushButton("执行攻击")
        self.btn_attack.clicked.connect(self.run_single_attack)
        control_layout.addWidget(self.btn_attack)

        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        self.figure = Figure(figsize=(10, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.pred_text = QTextEdit()
        self.pred_text.setFixedHeight(80)
        self.pred_text.setReadOnly(True)
        layout.addWidget(self.pred_text)

        self.current_index = None

    def load_random_image(self):
        idx = random.randint(0, self.total_test_samples - 1)
        self.current_index = idx
        x, y = get_single_image(self.loader, idx)
        self.current_x = x
        self.current_y = y

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(x.squeeze(), cmap='gray')
        ax.set_title(f"Original Image (label {y})")
        ax.axis('off')
        self.canvas.draw()
        self.pred_text.setText(f"已加载图像，标签: {y}。点击「执行攻击」生成对抗样本。")

    def run_single_attack(self):
        if self.current_x is None:
            QMessageBox.warning(self, "警告", "请先加载图像！")
            return

        attack_type = self.single_attack_type.currentText().lower()
        epsilon = self.single_eps.value()
        steps = self.single_steps.value()
        decay = self.single_decay.value()

        attack_kwargs = {}
        if attack_type == 'pgd':
            attack_kwargs['steps'] = steps
            attack_kwargs['alpha'] = epsilon * 0.25
            attack_kwargs['random_start'] = True
        elif attack_type == 'mi-fgsm': 
            attack_kwargs['steps'] = steps
            attack_kwargs['decay'] = decay
            attack_kwargs['random_start'] = True

        try:
            result = single_attack(
                self.model, self.current_x, self.current_y, self.device,
                attack_type=attack_type, epsilon=epsilon, **attack_kwargs
            )
        except Exception as e:
            QMessageBox.critical(self, "错误", f"攻击失败: {e}")
            return

        self.figure.clear()
        ax1 = self.figure.add_subplot(131)
        ax1.imshow(result['original'].squeeze(), cmap='gray')
        ax1.set_title("Original Image")
        ax1.axis('off')

        ax2 = self.figure.add_subplot(132)
        pert = result['perturbation'].squeeze()
        pert_display = (pert - pert.min()) / (pert.max() - pert.min() + 1e-8)
        ax2.imshow(pert_display, cmap='RdBu', vmin=0, vmax=1)
        ax2.set_title("Perturbation (amplified)")
        ax2.axis('off')

        ax3 = self.figure.add_subplot(133)
        ax3.imshow(result['adversarial'].squeeze(), cmap='gray')
        ax3.set_title("Adversarial Sample")
        ax3.axis('off')

        self.canvas.draw()

        orig_pred = result['orig_pred']
        orig_conf = result['orig_conf']
        adv_pred = result['adv_pred']
        adv_conf = result['adv_conf']
        label = result['orig_label']
        text = (f"原始预测: {orig_pred} (置信度 {orig_conf:.3f})  |  正确标签: {label}\n"
                f"对抗预测: {adv_pred} (置信度 {adv_conf:.3f})  |  攻击{'成功' if adv_pred != label else '失败'}")
        self.pred_text.setText(text)

    def setup_batch_tab(self):
        layout = QVBoxLayout(self.tab_batch)

        control_group = QGroupBox("批量评估控制")
        control_layout = QHBoxLayout()

        self.batch_attack_type = QComboBox()
        self.batch_attack_type.addItems(["FGSM", "PGD", "MI-FGSM"])
        control_layout.addWidget(QLabel("攻击方法:"))
        control_layout.addWidget(self.batch_attack_type)

        control_layout.addWidget(QLabel("防御方法:"))
        self.defense_combo = QComboBox()
        self.defense_combo.addItems(["无防御", "随机平滑"])
        control_layout.addWidget(self.defense_combo)

        control_layout.addWidget(QLabel("采样数:"))
        self.num_samples = QSpinBox()
        self.num_samples.setRange(10, 200)
        self.num_samples.setValue(50)
        control_layout.addWidget(self.num_samples)

        control_layout.addWidget(QLabel("平滑参数σ:"))
        self.sigma = QDoubleSpinBox()
        self.sigma.setRange(0.01, 0.5)
        self.sigma.setSingleStep(0.05)
        self.sigma.setValue(0.1)
        control_layout.addWidget(self.sigma)

        self.btn_batch_start = QPushButton("开始评估")
        self.btn_batch_start.clicked.connect(self.run_batch_eval)
        control_layout.addWidget(self.btn_batch_start)

        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.batch_result_tabs = QTabWidget()
        layout.addWidget(self.batch_result_tabs)

        self.fig_ex = Figure(figsize=(6, 4))
        self.canvas_ex = FigureCanvas(self.fig_ex)
        self.batch_result_tabs.addTab(self.canvas_ex, "攻击示例")

        self.fig_curve = Figure(figsize=(6, 4))
        self.canvas_curve = FigureCanvas(self.fig_curve)
        self.batch_result_tabs.addTab(self.canvas_curve, "鲁棒准确率")

        self.fig_diff = Figure(figsize=(6, 4))
        self.canvas_diff = FigureCanvas(self.fig_diff)
        self.batch_result_tabs.addTab(self.canvas_diff, "防御增益")

    def run_batch_eval(self):
        self.btn_batch_start.setEnabled(False)
        self.progress_bar.setRange(0, 0)

        attack_type = self.batch_attack_type.currentText().lower()
        defense_type = self.defense_combo.currentText()
        use_defense = (defense_type == "随机平滑")
        defense_params = {
            'num_samples': self.num_samples.value(),
            'sigma': self.sigma.value()
        } if use_defense else None
        attack_params = {
            'pgd_steps': 10,
            'alpha_ratio': 0.25,
            'random_start': True,
            'decay': 1.0
        }

        self.worker = BatchWorker(
            self.model, self.loader, self.device,
            attack_type, defense_params, attack_params
        )
        self.worker.finished.connect(self.on_batch_finished)
        self.worker.error.connect(self.on_batch_error)
        self.worker.start()

    def on_batch_finished(self, fig_ex, fig_curve, fig_diff):
        self._update_canvas(self.canvas_ex, fig_ex)
        self._update_canvas(self.canvas_curve, fig_curve)
        self._update_canvas(self.canvas_diff, fig_diff)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        self.btn_batch_start.setEnabled(True)
        QMessageBox.information(self, "完成", "批量评估完成！")

    def on_batch_error(self, err_msg):
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.btn_batch_start.setEnabled(True)
        QMessageBox.critical(self, "错误", f"评估失败: {err_msg}")

    def _update_canvas(self, canvas, fig):
        canvas.figure.clear()
        canvas.figure = fig
        canvas.draw()

    def closeEvent(self, event):
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()