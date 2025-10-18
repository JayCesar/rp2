from dataclasses import dataclass, field

from typing import Dict, Optional

@dataclass
class Redacao:

    id: int
    titulo: str
    texto: str
    nota_c1: int
    nota_final: int
    enunciado_titulo: str
    enunciado: str
    categoria: str

    gemini_nota_c1: Optional[int] = field(default=None)
    gemini_nota_final: Optional[int] = field(default=None)
    gemini_descricao: Optional[str] = field(default=None)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "titulo": self.titulo,
            "texto": self.texto,
            "nota_c1": self.nota_c1,
            "nota_final": self.nota_final,
            "enunciado_titulo": self.enunciado_titulo,
            "enunciado": self.enunciado,
            "categoria": self.categoria,

            "gemini_nota_c1": self.gemini_nota_c1,
            "gemini_nota_final": self.gemini_nota_final,
            "gemini_descricao": self.gemini_descricao
        }