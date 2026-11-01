"""Remedies for the six refused ATH feature groups; predicates stay authoritative."""
import pytest

from hornlab_mesher.config_parser import parse_text_config, ConfigError
from hornlab_mesher.config_builder import resolve_geometry
from hornlab_mesher.errors import UserRefusal

BASE = "OSSE = {\n L = 100\n r0 = 25\n a0 = 10\n a = 45\n}\nABEC.SimType = 2\n"


@pytest.mark.parametrize('extra, fix', [
    ('Mesh.SubdomainSlices = 3\n', 'Remove'),
    ('Throat.Profile = 3\n', 'Set “Throat.Profile” to 1'),
    ('Source.Contours = {\nx = 1\n}\n', 'Remove “Source.Contours”'),
    ('Source.Velocity = 2\n', 'Set “Source.Velocity” to 1'),
    ('Mesh.RearShape = 2\n', 'Set “Mesh.RearShape” to 1'),
    ('Mesh.Roundover = {\nRadius = 20\nSegments = 6\n}\nABEC.SimType = 1\n', 'Remove “Mesh.Roundover”'),
])
def test_ath_feature_refusal_names_a_supported_way_forward(extra, fix):
    with pytest.raises(ConfigError) as error:
        resolve_geometry(parse_text_config(BASE + extra))
    assert isinstance(error.value, UserRefusal)
    assert fix in error.value.next_step
    assert error.value.user_message and str(error.value)


def test_stretch_never_relabels_an_existing_refusal():
    from hornlab_mesher.throat_stretch import stretch_config_errors
    refusal = UserRefusal('Wall thickness', 'The wall folds.', 'Reduce Wall thickness.', diagnostic='original flip')
    @stretch_config_errors
    def fail(_config):
        raise refusal
    with pytest.raises(UserRefusal) as error:
        fail({'formula': 'OSSE', 's1': 1, 's2': 1})
    assert error.value is refusal
