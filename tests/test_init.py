"""Test the imports in the fridom/__init__.py file."""
import fridom


def test_type_checking_imports():
    """Test if the imports in the type checking block can be imported."""
    fridom._import_all()
    assert True
