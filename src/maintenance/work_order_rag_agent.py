"""
Agente Generativo y Prescriptivo de Mantenimiento (SAP PM / ISO 13374 / SAE J1939 RAG).
Convierte scores de anomalía y residuos de sensores en Órdenes de Trabajo técnicas
estructuradas para mecánicos de taller y planificadores de mina.
Cumple con la norma ISO 13374 (Bloques de Diagnóstico y Generación de Recomendaciones).
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class DiagnosticTroubleCode:
    spn: int
    fmi: int
    system: str
    description: str
    subsystem_label: str
    oem_parts: List[Dict[str, str]]
    prescriptive_steps: List[str]


class MaintenanceWorkOrderAgent:
    """
    Agente RAG de Mantenimiento Predictivo para Gran Minería (CAEX).
    Realiza atribución de causa raíz mediante análisis de residuos y consulta
    la base de conocimiento de códigos SAE J1939 y catálogos OEM para redactar
    fichas técnicas de SAP PM listas para taller.
    """

    # Líneas base de referencia nominal (Especificaciones CAEX Komatsu 930E / Cat 797F)
    NOMINAL_BASELINES: Dict[str, Tuple[float, float]] = {
        'engine_temp': (85.0, 3.0),
        'oil_pressure': (45.0, 2.0),
        'vibration_level': (2.5, 0.3),
        'rpm': (1800.0, 50.0),
        'fuel_consumption': (35.0, 2.0),
        'coolant_temp': (78.0, 2.5),
        'hydraulic_pressure': (3200.0, 100.0),
    }

    # Base de conocimiento técnico industrial SAE J1939 / Catálogos OEM
    KNOWLEDGE_BASE: Dict[str, DiagnosticTroubleCode] = {
        "COOLING_CIRCUIT": DiagnosticTroubleCode(
            spn=110,
            fmi=0,
            system="CIRCUITO DE REFRIGERACIÓN Y DISIPACIÓN TÉRMICA",
            description="Sobrecalentamiento severo de motor con gradiente anormal en refrigerante",
            subsystem_label="Sistema Térmico / Culata / Bomba de Agua",
            oem_parts=[
                {"part_no": "KM-98210-A", "desc": "Kit sellos bomba refrigerante Komatsu 930E", "qty": "1 un"},
                {"part_no": "CAT-248-5513", "desc": "Termostato dual Cummins QSK78 / Cat 797F", "qty": "2 un"},
                {"part_no": "CH-RAD-600", "desc": "Refrigerante etilenglicol 50/50 OAT", "qty": "40 L"}
            ],
            prescriptive_steps=[
                "Aislar camión en bahía de mantenimiento mecánico (Lockout/Tagout).",
                "Permitir enfriamiento pasivo antes de abrir circuito presurizado.",
                "Inspeccionar visualmente pérdidas de líquido en sellos de culata y mangueras de retorno.",
                "Escanear con cámara termográfica el radiador frontal para detectar tubos tapados.",
                "Reemplazar kit de sellos o termostato según prueba hidrostática de presión a 25 PSI."
            ]
        ),
        "LUBRICATION_CIRCUIT": DiagnosticTroubleCode(
            spn=100,
            fmi=1,
            system="CIRCUITO DE LUBRICACIÓN Y ALTA PRESIÓN",
            description="Caída crítica de presión de aceite con pérdida de viscosidad dinámica",
            subsystem_label="Bomba de Aceite / Filtros / Cojinetes de Bancada",
            oem_parts=[
                {"part_no": "KM-31420-L", "desc": "Bomba de engranajes aceite alta presión", "qty": "1 un"},
                {"part_no": "CAT-1R-1808", "desc": "Filtro microfibra sintética bypass de lubricación", "qty": "4 un"},
                {"part_no": "MOB-DEL-15W40", "desc": "Aceite sintético para trabajo pesado Mobil Delvac 1", "qty": "200 L"}
            ],
            prescriptive_steps=[
                "Detener inmediatamente el motor para prevenir agarrotamiento de cigüeñal (Metal-to-Metal).",
                "Drenar muestra de aceite de 500 ml para espectrometría de partículas ferrosas y bronce (ASTM D5185).",
                "Inspeccionar el cartucho del filtro de aceite con imán en busca de viruta de metales blancos.",
                "Verificar la válvula reguladora de presión de la galería principal de lubricación.",
                "Realizar prueba de torque en pernos de bancada de bielas si la muestra presenta cobre."
            ]
        ),
        "MECHANICAL_POWERTRAIN": DiagnosticTroubleCode(
            spn=190,
            fmi=0,
            system="TREN DE FUERZA Y VIBRACIÓN DINÁMICA",
            description="Vibración armónica excesiva y desbalanceo cinemático en eje motor/transmisión",
            subsystem_label="Damper Torsional / Rodamientos / Cardán Principal",
            oem_parts=[
                {"part_no": "KM-67500-D", "desc": "Damper viscoso de vibración torsional cigüeñal", "qty": "1 un"},
                {"part_no": "SKF-23244-CC", "desc": "Rodamiento oscilante de rodillos SKF Explorer", "qty": "2 un"},
                {"part_no": "SPI-1810-U", "desc": "Cruceta Spicer serie 1810 para cardán de acarreo", "qty": "2 un"}
            ],
            prescriptive_steps=[
                "Posicionar camión sobre fosa y bloquear ruedas motrices con cuñas de faena.",
                "Efectuar análisis espectral FFT en bahía con analizador de vibraciones portátil.",
                "Verificar juego axial y radial en crucetas y cojinete de soporte central.",
                "Inspeccionar el damper de vibración torsional por fuga de fluido de silicona.",
                "Reemplazar componentes con desalineación angular superior a 0.05 mm."
            ]
        ),
        "HYDRAULIC_HOIST": DiagnosticTroubleCode(
            spn=746,
            fmi=4,
            system="SISTEMA HIDRÁULICO DE TOLVA Y DIRECCIÓN",
            description="Pérdida de presión en cilindros de volteo o circuito asistido de dirección",
            subsystem_label="Bomba de Pistones Axiales / Múltiple Proporcional",
            oem_parts=[
                {"part_no": "KM-HYD-9901", "desc": "Válvula de control direccional Rexroth 350 bar", "qty": "1 un"},
                {"part_no": "CAT-4SH-16", "desc": "Manguera blindada 4-mallas espiral de alta presión", "qty": "2 un"},
                {"part_no": "HYD-ISO-VG68", "desc": "Fluido hidráulico antidesgaste ISO VG 68", "qty": "150 L"}
            ],
            prescriptive_steps=[
                "Asegurar la tolva del camión con los cables mecánicos de bloqueo de seguridad.",
                "Conectar manómetros digitales a los puertos de test rápido del bloque hidráulico.",
                "Verificar presión de alivio principal a velocidad nominal de motor (debe superar 210 bar).",
                "Inspeccionar vástagos de cilindros telescópicos por rayaduras o desgaste de sellos chevron.",
                "Purgar aire atrapado en circuito de servo-dirección tras intervención."
            ]
        ),
        "FUEL_SYSTEM": DiagnosticTroubleCode(
            spn=94,
            fmi=3,
            system="SISTEMA DE INYECCIÓN Y SUMINISTRO DE COMBUSTIBLE",
            description="Anomalía de consumo y presión en riel común de inyectores diésel (HPCR)",
            subsystem_label="Inyectores Electrónicos / Bomba Common Rail",
            oem_parts=[
                {"part_no": "BOS-CR-0445", "desc": "Inyector Common Rail Bosch de alta presión", "qty": "2 un"},
                {"part_no": "CAT-1R-0749", "desc": "Filtro separador de agua primario combustible", "qty": "2 un"}
            ],
            prescriptive_steps=[
                "Ejecutar prueba de corte de cilindros electrónico con software de diagnóstico.",
                "Medir retorno de caudal de inyectores para detectar toberas erosionadas.",
                "Revisar presencia de agua o micro-partículas en el filtro separador primario.",
                "Reemplazar inyectores del banco defectuoso y reprogramar códigos de compensación IMA."
            ]
        )
    }

    def diagnose_root_cause(self, sensor_readings: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
        """
        Calcula las desviaciones z-score de cada sensor respecto a su firma de calibración
        y atribuye la falla al subsistema con mayor anomalía física.
        """
        z_scores = {}
        for sensor, (mu, sigma) in self.NOMINAL_BASELINES.items():
            if sensor in sensor_readings:
                z_scores[sensor] = (sensor_readings[sensor] - mu) / sigma

        # Puntuación por subsistema
        subsystem_scores = {
            "COOLING_CIRCUIT": max(z_scores.get('engine_temp', 0.0), z_scores.get('coolant_temp', 0.0)),
            # Para presión de aceite, una caída (z negativo) es lo crítico:
            "LUBRICATION_CIRCUIT": -min(z_scores.get('oil_pressure', 0.0), 0.0) * 1.5,
            "MECHANICAL_POWERTRAIN": max(z_scores.get('vibration_level', 0.0), 0.0) * 1.3,
            # Para hidráulica, caídas severas:
            "HYDRAULIC_HOIST": -min(z_scores.get('hydraulic_pressure', 0.0), 0.0),
            "FUEL_SYSTEM": abs(z_scores.get('fuel_consumption', 0.0))
        }

        # Identificar subsistema dominante
        root_subsystem = max(subsystem_scores, key=subsystem_scores.get)
        return root_subsystem, z_scores

    def generate_work_order(
        self,
        equipment_id: str,
        sensor_readings: Dict[str, float],
        health_score: float,
        cbm_decision: Dict[str, Any],
        weibull_rul: Dict[str, Any],
        fleet_model: str = "KOMATSU 930E-4SE"
    ) -> Dict[str, Any]:
        """
        Genera una Orden de Trabajo completa compatible con SAP PM / ISO 13374.
        """
        root_subsystem, z_scores = self.diagnose_root_cause(sensor_readings)
        dtc = self.KNOWLEDGE_BASE.get(root_subsystem, self.KNOWLEDGE_BASE["COOLING_CIRCUIT"])

        urgency = cbm_decision.get("urgency", "CRITICAL")
        priority_code = 1 if urgency == "CRITICAL" else (2 if urgency == "WARNING" else 4)
        priority_label = "1 - ALTA (PARADA PROGRAMADA INMEDIATA)" if priority_code == 1 else "2 - MEDIA (SIGUIENTE TURNO)"

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        rul_p10 = weibull_rul.get("rul_p10_conservative", 0.0)
        p_crit = cbm_decision.get("p_critical", 0.0) * 100.0

        # Resumen de diagnóstico en lenguaje natural industrial
        top_anomalies = [
            f"{s}: {sensor_readings[s]:.1f} ({z:+.1f}σ)"
            for s, z in sorted(z_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        ]
        anomaly_summary = ", ".join(top_anomalies)

        # Formato texto estructurado SAP PM (Notification PM01 / PM02)
        sap_pm_card = (
            f"================================================================================\n"
            f"ORDEN DE TRABAJO AUTOMATIZADA - SISTEMA CBM (SAP PM / ISO 13374)\n"
            f"EQUIPO: {equipment_id} | FLOTA: {fleet_model} | FECHA: {now_str}\n"
            f"PRIORIDAD: {priority_label}\n"
            f"================================================================================\n"
            f"DIAGNÓSTICO ANALÍTICO:\n"
            f"- Health Score: {health_score:.1f} / 100 | Riesgo Crítico: {p_crit:.2f}%\n"
            f"- RUL Estimado (P10): {rul_p10:.1f} ciclos operacionales\n"
            f"- Subsistema Comprometido: {dtc.system}\n"
            f"- Causa Raíz Detectada: {dtc.description} [{anomaly_summary}]\n"
            f"- Código de Falla SAE J1939: SPN {dtc.spn} / FMI {dtc.fmi:02d} ({dtc.subsystem_label})\n\n"
            f"PLAN DE ACCIÓN PRESCRIPTIVO PARA TALLER:\n"
        )
        for idx, step in enumerate(dtc.prescriptive_steps, 1):
            sap_pm_card += f"{idx}. {step}\n"

        sap_pm_card += f"\nLISTA DE REPUESTOS REQUERIDOS (BOM / OEM):\n"
        for p in dtc.oem_parts:
            sap_pm_card += f"- Part #{p['part_no']}: {p['desc']} (Cant: {p['qty']})\n"

        sap_pm_card += f"================================================================================"

        return {
            "work_order_id": f"WO-{equipment_id}-{datetime.now().strftime('%Y%m%d%H%M')}",
            "equipment_id": equipment_id,
            "fleet_model": fleet_model,
            "created_at": now_str,
            "priority": priority_code,
            "priority_label": priority_label,
            "subsystem": root_subsystem,
            "subsystem_name": dtc.system,
            "sae_j1939_spn": dtc.spn,
            "sae_j1939_fmi": dtc.fmi,
            "fault_code_label": f"SPN {dtc.spn} FMI {dtc.fmi}",
            "root_cause": dtc.description,
            "health_score": round(health_score, 1),
            "rul_p10": rul_p10,
            "prescriptive_action_plan": dtc.prescriptive_steps,
            "required_parts": dtc.oem_parts,
            "sap_pm_formatted_card": sap_pm_card
        }
