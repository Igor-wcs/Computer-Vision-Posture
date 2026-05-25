import unittest
from main import PostureDetectorApp

class TestPostureLogic(unittest.TestCase):
    
    def test_good_posture_logic(self):
        """Testa se a lógica identifica corretamente uma boa postura."""
        # Coordenadas: (x, y)
        nose = (0.5, 0.4)
        l_shoulder = (0.4, 0.6)
        r_shoulder = (0.6, 0.6)
        
        status = PostureDetectorApp.classify_posture(nose, l_shoulder, r_shoulder)
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
