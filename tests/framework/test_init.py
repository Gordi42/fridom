"""Test the imports in the fridom/framework/__init__.py file."""
import fridom.framework as fr


def test_type_checking_imports():
    """Test if the imports in the type checking block can be imported."""
    fr._import_all()
    assert True
