# Prefijo uuid
ID_PREFIX = "TK-"

# Set de stopwords en español comunes
STOPWORDS = {"de", "la", "el", "y", "o", "que"}

# Frases que indican que RAG no sabe o no tiene información
RAG_NEGATIVE_PHRASES = [
    "no contiene información",
    "no se encontró información",
    "no incluye información",
    "no puedo responder",
    "desconozco"
]

# Lista de ejemplos de consultas para el Helpdesk
HELPDESK_EXAMPLES = [
    # Acceso y credenciales
    "No puedo resetear mi contraseña",
    "Mi cuenta está bloqueada",
    "Olvidé mi usuario",
    "No puedo acceder a mi cuenta",
    "No recuerdo mi contraseña",
    "Cuenta suspendida",
    
    # Errores del sistema / bugs
    "Error 500 en la aplicación",
    "Error 404 - Página no encontrada",
    "La página de inicio no carga",
    "Se congela la aplicación al iniciar sesión",
    "La aplicación va muy lenta",
    "Problemas de visualización en el dashboard",
    "Errores de sincronización de datos",
    "Problemas de rendimiento de la aplicación",

    # Suscripciones y pagos / gestión de planes
    "¿Cómo cancelo mi suscripción?",
    "Me cobraron dos veces el mismo mes",
    "No recibí mi factura",
    "Quiero cambiar mi plan de suscripción",
    "¿Cómo actualizo mi plan?",
    "¿Puedo obtener un reembolso?",
    
    # Funcionalidades y preguntas frecuentes
    "¿Cómo exporto mis datos?",
    "¿Dónde encuentro el historial de pedidos?",
    "Quiero configurar notificaciones por correo",
    "¿Cómo agrego un nuevo usuario al equipo?",
    "Quiero cambiar el idioma de la interfaz",
    "Subir archivos PDF o CSV",
    "Cómo organizar mis datos con carpetas y etiquetas",
    "Buscar información por nombre o contenido",
    
    # Integraciones y APIs
    "No puedo conectar con Google Drive",
    "Error al sincronizar con Outlook",
    
    # Conectividad y red
    "No tengo conexión a internet",
    "Problemas con firewall corporativo",
    "No puedo acceder desde la red de mi empresa",
    
    # Seguridad
    "Cómo habilitar autenticación de dos factores",
    "Reportar actividad sospechosa",
    "Problemas de seguridad en mi cuenta",
    
    # Contacto de emergencia / soporte urgente
    "Necesito soporte técnico urgente",
    "Reportar un problema de seguridad",
    
    # Primeros pasos / onboarding
    "Cómo configurar mi cuenta inicial",
    "Cómo navegar en el dashboard principal",
    "Cómo ajustar las preferencias de notificación",
    
    # Problemas generales y otros
    "Recibo mensajes de error desconocidos",
    "No puedo subir archivos adjuntos",
    "La aplicación no envía notificaciones",
]

