"""Document and corpus schemas."""

from typing import Optional, List
from pydantic import BaseModel, Field
from enum import Enum


class DocumentType(str, Enum):
    """Document classification."""
    ALARM_MANUAL = "alarm_manual"
    MANUAL = "manual"
    SOP = "sop"
    TROUBLESHOOTING = "troubleshooting"
    MAINTENANCE = "maintenance"
    FAQ = "faq"
    ENGINEER_NOTE = "engineer_note"


class TrustLevel(str, Enum):
    """Trust level classification."""
    OFFICIAL_ALARM = 100
    OFFICIAL_MANUAL = 95
    OFFICIAL_SOP = 90
    OFFICIAL_TROUBLESHOOTING = 85
    MAINTENANCE_HISTORY = 70
    FAQ = 60
    ENGINEER_NOTE = 50


class DocumentChunk(BaseModel):
    """Document chunk with metadata."""
    chunk_id: str

    # Content
    text: str
    section: Optional[str] = None

    # Document metadata
    machine: Optional[str] = None
    manufacturer: Optional[str] = None
    document_type: DocumentType
    document_name: str
    source: str

    # System/Component
    system: Optional[str] = None
    component: Optional[str] = None

    # Alarm
    alarm_code: Optional[str] = None
    symptoms: List[str] = []

    # Version
    software_version: Optional[str] = None
    firmware_version: Optional[str] = None

    # Metadata
    language: str = "ko"
    trust_level: int = Field(default=60, ge=0, le=100)
    tags: List[str] = []

    # Timestamps
    created_at: Optional[float] = None
    updated_at: Optional[float] = None


class AlarmEntry(BaseModel):
    """Alarm code entry."""
    alarm_code: str
    normalized_code: str  # Normalized format
    title: str
    symptoms: List[str]
    possible_causes: List[str]
    inspection_steps: List[str]
    normal_ranges: dict
    document_chunk_id: str


class MaintenanceRecord(BaseModel):
    """Maintenance history record."""
    record_id: str
    timestamp: float
    machine: str
    alarm_code: Optional[str] = None
    component: Optional[str] = None
    symptom: str
    action_taken: str
    result: str
    engineer_note: Optional[str] = None
    severity: Optional[str] = None  # "low", "medium", "high"
