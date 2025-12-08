"""
Script para crear un video MP4 a partir de los frames guardados en video_frames/
"""
from pathlib import Path
import sys

def create_video_from_frames(frames_dir="video_frames", output_path="optimization_video.mp4", fps=10):
    """
    Crea un video MP4 a partir de los frames guardados.
    
    Args:
        frames_dir: Directorio con los frames
        output_path: Ruta del archivo de video de salida
        fps: Frames por segundo del video
    """
    try:
        import imageio
    except ImportError:
        print("ERROR: imageio no está instalado.")
        print("Instala con: pip install imageio imageio-ffmpeg")
        sys.exit(1)
    
    frames_path = Path(frames_dir)
    
    if not frames_path.exists():
        print(f"ERROR: No existe el directorio {frames_dir}")
        sys.exit(1)
    
    # Obtener todos los frames ordenados
    frame_files = sorted(frames_path.glob("frame_*.png"))
    
    if not frame_files:
        print(f"ERROR: No se encontraron frames en {frames_dir}")
        sys.exit(1)
    
    print(f"Encontrados {len(frame_files)} frames")
    print(f"Creando video a {fps} fps...")
    
    try:
        # Leer frames y redimensionar al mismo tamaño
        import numpy as np
        try:
            from PIL import Image
        except ImportError:
            print("ERROR: Pillow no está instalado.")
            print("Instala con: pip install Pillow")
            sys.exit(1)
        
        frames = []
        target_size = None
        
        print("  Leyendo y redimensionando frames...")
        for i, frame_file in enumerate(frame_files):
            img = Image.open(frame_file)
            
            # Determinar el tamaño objetivo del primer frame
            if target_size is None:
                # Asegurar que las dimensiones sean pares (divisibles por 2) para libx264
                width, height = img.size
                # Redondear al número par más cercano
                width = (width // 2) * 2
                height = (height // 2) * 2
                target_size = (width, height)
                print(f"  Tamaño objetivo (ajustado a par): {target_size[0]}x{target_size[1]}")
            
            # Redimensionar al tamaño objetivo (siempre, para asegurar consistencia)
            if img.size != target_size:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Convertir a array numpy
            frames.append(np.array(img))
            
            if (i + 1) % 50 == 0:
                print(f"  Procesados {i + 1}/{len(frame_files)} frames...")
        
        print(f"  Total de frames procesados: {len(frames)}")
        print("  Creando video...")
        
        # Verificar que todas las imágenes tengan el mismo tamaño
        first_shape = frames[0].shape
        for i, frame in enumerate(frames):
            if frame.shape != first_shape:
                print(f"  ADVERTENCIA: Frame {i} tiene tamaño diferente: {frame.shape} vs {first_shape}")
        
        # Crear video MP4 (compatible con WhatsApp)
        output_path = Path(output_path)
        # Crear video con parámetros compatibles (dimensiones ya son pares)
        imageio.mimsave(
            output_path, 
            frames, 
            fps=fps, 
            codec='libx264', 
            quality=8,
            pixelformat='yuv420p'  # Formato compatible con WhatsApp
        )
        
        print(f"\n✅ Video creado exitosamente: {output_path.absolute()}")
        print(f"   El video es compatible con WhatsApp (MP4)")
        print(f"   Tamaño: {output_path.stat().st_size / (1024*1024):.2f} MB")
        
        return str(output_path.absolute())
        
    except Exception as e:
        print(f"ERROR al crear video: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Crear video MP4 desde frames")
    parser.add_argument("--frames-dir", default="video_frames", 
                       help="Directorio con los frames (default: video_frames)")
    parser.add_argument("--output", default="optimization_video.mp4",
                       help="Archivo de salida (default: optimization_video.mp4)")
    parser.add_argument("--fps", type=int, default=10,
                       help="Frames por segundo (default: 10)")
    
    args = parser.parse_args()
    
    create_video_from_frames(args.frames_dir, args.output, args.fps)

