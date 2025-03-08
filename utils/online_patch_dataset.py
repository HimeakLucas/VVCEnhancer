import os
import random
from PIL import Image
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

class OnlinePatchDataset(Dataset):
    def __init__(self, original_dir, processed_dir, patch_size=64, transform=None):
        """
        Dataset para extração de patches de imagens originais e processadas,
        utilizando os diretórios fornecidos.
        
        Args:
            original_dir (str): Diretório das imagens originais.
            processed_dir (str): Diretório das imagens processadas.
            patch_size (int): Tamanho dos patches extraídos.
            transform (callable, optional): Transformação a ser aplicada aos patches.
        """
        self.patch_size = patch_size
        self.transform = transform
        self.pairs = []

        # Diretórios de imagens originais e processadas
        self.original_dir = original_dir
        self.processed_dir = processed_dir

        # Mapeia frames de cada vídeo
        self.original_videos = self._map_video_frames(self.original_dir)
        self.processed_videos = self._map_video_frames(self.processed_dir)

        # Valida e cria pares (ignorando vídeos sem correspondência)
        self._validate_directories()
        self._create_pairs()

    def _map_video_frames(self, root_dir):
        #Mapeia a estrutura de diretórios para vídeos e seus frames PNG.
        #Retorna um dicionário onde a chave é o nome do vídeo e o valor é uma lista de frames. 
        #Assume que o diretório raiz contém pastas para cada vídeo, com frames numerados. 
        video_map = {}
        for video_name in os.listdir(root_dir):
            video_path = os.path.join(root_dir, video_name)
            if os.path.isdir(video_path):
                frames_info = []
                for fname in sorted(os.listdir(video_path)):
                    if fname.endswith('.png'):
                        frame_num = self._extract_frame_number(fname)
                        frames_info.append({
                            'path': os.path.join(video_path, fname),
                            'number': frame_num
                        })
                if frames_info:
                    video_map[video_name] = frames_info
        return video_map

    def _extract_frame_number(self, filename):
        """
        Extrai o número do frame a partir do nome do arquivo.
        Exemplo: 'frame_0001.png' -> 1.
        """
        base = os.path.splitext(filename)[0]
        return int(base.split('_')[-1])

    def _validate_directories(self):
        """
        Garante que cada pasta de vídeo em processed_videos tenha um correspondente
        em original_videos. Se não houver correspondência, emite um aviso e remove o vídeo.
        """
        keys_to_remove = []
        for processed_video in self.processed_videos:
            if processed_video not in self.original_videos:
                print(f"Warning: Vídeo processado '{processed_video}' não tem correspondente em '{self.original_dir}'. Ignorando-o.")
                keys_to_remove.append(processed_video)
        for key in keys_to_remove:
            del self.processed_videos[key]

    def _create_pairs(self):
        """Cria pares (caminho_original, caminho_processado) para cada frame correspondente."""
        for video_name, proc_frames in self.processed_videos.items():
            original_frames = self.original_videos.get(video_name, [])
            # Mapeia os frames originais pelo número
            original_frame_map = {f['number']: f['path'] for f in original_frames}
            for proc_frame in proc_frames:
                frame_num = proc_frame['number']
                if frame_num in original_frame_map:
                    self.pairs.append((
                        original_frame_map[frame_num],
                        proc_frame['path']
                    ))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        orig_path, proc_path = self.pairs[idx]

        orig_img = Image.open(orig_path).convert('L')
        proc_img = Image.open(proc_path).convert('L')

        if orig_img.size != proc_img.size:
            raise ValueError(f"Tamanho diferente entre {orig_path} e {proc_path}")

        # Seleciona um patch com conteúdo relevante usando detecção de bordas
        patch_box = self._select_patch(np.array(orig_img))

        orig_patch = orig_img.crop(patch_box)
        proc_patch = proc_img.crop(patch_box)

        if self.transform:
            orig_patch = self.transform(orig_patch)
            proc_patch = self.transform(proc_patch)

        return orig_patch, proc_patch

    def _select_patch(self, img_np):
        """
        Seleciona um patch de tamanho self.patch_size x self.patch_size que contenha
        conteúdo relevante (baseado em detecção de bordas).
        """
        edges = cv2.Canny(img_np, 50, 150)
        h, w = img_np.shape

        for _ in range(100):
            x = random.randint(0, w - self.patch_size)
            y = random.randint(0, h - self.patch_size)
            edge_patch = edges[y:y + self.patch_size, x:x + self.patch_size]
            if edge_patch.mean() > 5:  # Threshold para "conteúdo interessante"
                return (x, y, x + self.patch_size, y + self.patch_size)

        # Se não encontrar patch com bordas significativas, retorna um patch aleatório
        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size)
        return (x, y, x + self.patch_size, y + self.patch_size)

if __name__ == "__main__":
    # Exemplo: criando dataset para treino com QP = "qp47"
    train_dataset = OnlinePatchDataset(
        original_dir="../data/frames_y/train/original",
        processed_dir="../data/frames_y/train/qp63",
        patch_size=64,
        transform=transforms.ToTensor()
    )
    print("Número de pares de treino:", len(train_dataset))

    # Exemplo: criando dataset para validação
    val_dataset = OnlinePatchDataset(
        original_dir="../data/frames_y/val/original",
        processed_dir="../data/frames_y/val/qp63",
        patch_size=64,
        transform=transforms.ToTensor()
    )
    print("Número de pares de validação:", len(val_dataset))