from rag_module.pipeline import RAGPipeline
import sys


def main():
    """Demo interactivo del módulo RAG."""
    try:
        print("\n" + "="*60)
        print("🎬 DEMO RAG - Búsqueda de Películas y Series")
        print("="*60)
        print("\nInicializando sistema...")
        
        rag = RAGPipeline()
        
        print("✅ Sistema inicializado correctamente")
        print("\n📝 Instrucciones:")
        print("  • Escribe una pregunta sobre películas o series")
        print("  • Escribe 'salir' para terminar")
        print("  • Presiona Ctrl+C para interrumpir\n")
        print("-"*60)
        
        while True:
            try:
                pregunta = input("\n🔍 Pregunta: ").strip()
                
                # Validar entrada
                if not pregunta:
                    print("⚠️  Por favor, escribe una pregunta válida")
                    continue
                
                # Verificar si quiere salir
                if pregunta.lower() in ["salir", "exit", "quit"]:
                    print("\n✋ ¡Hasta luego!\n")
                    break
                
                print("\n⏳ Buscando información y generando respuesta...")
                respuesta = rag.query(pregunta)
                
                print("\n" + "─"*60)
                print("💡 RESPUESTA:")
                print("─"*60)
                print(respuesta)
                print("─"*60)
            
            except KeyboardInterrupt:
                print("\n\n✋ Operación cancelada por el usuario")
                break
            except Exception as e:
                print(f"\n⚠️  Error procesando pregunta: {e}")
                print("   Intenta de nuevo con otra pregunta")
    
    except KeyboardInterrupt:
        print("\n\n✋ Aplicación interrumpida")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error fatal: {e}")
        print("\n⚠️  Soluciones:")
        print("  1. Ollama no está corriendo:")
        print("     $ ollama serve")

        print("\n  2. Verifica que NeuralRetriever esté disponible")
        sys.exit(1)


if __name__ == "__main__":
    main()