"""Actionable application-specific exceptions."""


class FaceApplicationError(RuntimeError):
    """Base class for expected, user-facing failures."""


class ModelError(FaceApplicationError):
    """A recognition or detection model cannot be used."""


class ModelMissingError(ModelError):
    """A required local model file is absent."""


class ModelIntegrityError(ModelError):
    """A downloaded model did not pass integrity checks."""


class DetectorError(FaceApplicationError):
    """Face detector initialization or inference failed."""


class RecognitionError(FaceApplicationError):
    """Embedding creation or comparison failed."""


class LandmarksRequiredError(RecognitionError):
    """SFace was requested for a detection without YuNet landmarks."""


class DatabaseError(FaceApplicationError):
    """Persistent local embedding storage is invalid or unavailable."""


class DuplicatePersonError(DatabaseError):
    """A name already maps to an existing person without append consent."""


class AmbiguousPersonError(DatabaseError):
    """A name maps to more than one registered person."""


class EnrollmentError(FaceApplicationError):
    """A person cannot be safely enrolled from the requested sample."""


class NoFaceDetectedError(EnrollmentError):
    """An enrollment sample does not contain a usable face."""


class MultipleFacesDetectedError(EnrollmentError):
    """An enrollment sample contains more than one face."""


class MediaError(FaceApplicationError):
    """Input media or output media cannot be processed."""


class CameraOpenError(MediaError):
    """A requested camera cannot be opened."""
