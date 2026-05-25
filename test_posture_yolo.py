import unittest
import numpy as np
from main_yolov26 import PostureDetectorAppYOLO

class TestPostureYOLOLogic(unittest.TestCase):
    
    def setUp(self):
        self.app = PostureDetectorAppYOLO()
        # Mock de calibração
        self.app.base_head_dist = 0.2
        self.app.base_shoulder_width = 0.3
        self.app.calibrated = True

    def test_good_posture_logic(self):
        """Testa se a lógica identifica corretamente uma boa postura no YOLO."""
        nose = np.array([0.5, 0.4])
        l_shoulder = np.array([0.4, 0.6])
        r_shoulder = np.array([0.6, 0.6])
        head_dist = 0.2
        shoulder_width = 0.2
        
        # Ajustando para os valores de setup
        status = self.app.classify_posture_yolo(nose, l_shoulder, r_shoulder, 0.2, 0.3)
        self.assertEqual(status, "Boa Postura")

    def test_head_down_logic(self):
        """Testa se a lógica identifica 'Cabeça Muito Baixa'."""
        nose = np.array([0.5, 0.55])
        l_shoulder = np.array([0.4, 0.6])
        r_shoulder = np.array([0.6, 0.6])
        
        # head_dist muito pequeno (0.05 vs base 0.2)
        status = self.app.classify_posture_yolo(nose, l_shoulder, r_shoulder, 0.05, 0.3)
        self.assertEqual(status, "Cabeça Muito Baixa!")

    def test_shoulder_misalignment_logic(self):
        """Testa se a lógica identifica 'Ombros Desalinhados'."""
        nose = np.array([0.5, 0.4])
        l_shoulder = np.array([0.4, 0.5])
        r_shoulder = np.array([0.6, 0.6])
        
        status = self.app.classify_posture_yolo(nose, l_shoulder, r_shoulder, 0.1, 0.3)
        self.assertEqual(status, "Ombros Desalinhados!")

    def test_forward_lean_logic(self):
        """Testa se a lógica identifica 'Inclinado para Frente' (ombros largos)."""
        nose = np.array([0.5, 0.4])
        l_shoulder = np.array([0.3, 0.6])
        r_shoulder = np.array([0.7, 0.6])
        
        # shoulder_width = 0.4 (maior que base 0.3 * 1.2 = 0.36)
        status = self.app.classify_posture_yolo(nose, l_shoulder, r_shoulder, 0.2, 0.4)
        self.assertEqual(status, "Inclinado para Frente!")

if __name__ == '__main__':
    # Usamos um try/except para evitar erro de inicialização do CustomTkinter em ambientes sem display
    try:
        unittest.main()
    except Exception as e:
        print(f"Erro ao rodar testes (provavelmente falta de display): {e}")
