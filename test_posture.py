import unittest
from main import PostureDetectorApp

class TestPostureLogic(unittest.TestCase):
    
    def test_good_posture_logic(self):
        """Testa se a lógica identifica corretamente uma boa postura."""
        app = PostureDetectorApp()
        app.base_head_dist = 0.2
        app.base_shoulder_width = 0.3
        app.base_z_depth = 0.0
        app.calibrated = True

        # Coordenadas: (x, y, z)
        nose = (0.5, 0.4, 0.0)
        l_shoulder = (0.4, 0.6, 0.0)
        r_shoulder = (0.6, 0.6, 0.0)
        
        status = app.classify_enhanced_posture(nose, l_shoulder, r_shoulder, 0.2, 0.2, 0.0)
        self.assertEqual(status, "Boa Postura")

    def test_head_down_logic(self):
        """Testa se a lógica identifica 'Cabeça Baixa'."""
        # Nariz próximo da linha dos ombros (Y alto)
        nose = (0.5, 0.55)
        l_shoulder = (0.4, 0.6)
        r_shoulder = (0.6, 0.6)
        
        status = PostureDetectorApp.classify_posture(nose, l_shoulder, r_shoulder)
        self.assertEqual(status, "Cabeça Baixa!")

    def test_shoulder_misalignment_logic(self):
        """Testa se a lógica identifica 'Ombros Desalinhados'."""
        # Ombro esquerdo mais alto que o direito
        nose = (0.5, 0.4)
        l_shoulder = (0.4, 0.5)
        r_shoulder = (0.6, 0.6)
        
        status = PostureDetectorApp.classify_posture(nose, l_shoulder, r_shoulder)
        self.assertEqual(status, "Ombros Desalinhados!")

if __name__ == '__main__':
    unittest.main()
