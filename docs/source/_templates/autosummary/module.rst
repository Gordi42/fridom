{{ fullname | item_name | escape | underline}}

Module: :py:mod:`{{ fullname }}`

.. ==================================================
..  Template for modules that are __init__.py files
.. ==================================================

{% if fullname | is_init %}

.. --------------------------------------------------
..  Submodules Section
.. --------------------------------------------------

{% block submodules %}
{% set all_subs = fullname | get_submodules %}
{% if all_subs %}

.. rubric:: {{ _('Submodules') }}

.. list-table::
   :widths: 10 90

   {% for item in all_subs %}
   * - :py:mod:`{{ item | item_name }} <{{ item }}>`
     - {{ item | doc_summary_module }}
   {%- endfor %}

..
   .. autosummary::
      :toctree:
      :recursive:
      {% for item in all_subs %}
      {{ item }}
      {%- endfor %}

.. toctree::
   :includehidden:
   :hidden:

   {% for item in all_subs %}
   {{ item }}
   {%- endfor %}

{% endif %}
{% endblock %}

.. --------------------------------------------------
..  Classes, Functions, and Variables Section
.. --------------------------------------------------

{% block imports %}
{% set all_imports = fullname | get_imports %}
{% if all_imports %}

.. rubric:: {{ _('Classes, Functions, and Variables') }}

{% set imp_by_parent = all_imports | split_by_parent %}
{% set multiple_parents = imp_by_parent | count > 1 %}

{% for parent, childrens in imp_by_parent %}
.. currentmodule:: {{ parent }}

.. autosummary::
   :toctree:
   :recursive:
   {% for item in childrens %}
   {{ item }}
   {%- endfor %}

{% endfor %}
{% endif %}
{% endblock %}

.. automodule:: {{ fullname }}
   :no-members:
   :no-inherited-members:
   :no-special-members:
   :no-private-members:
   :undoc-members:

.. ==================================================
..  Template for modules that are no __init__.py files
.. ==================================================

{% else %}

.. currentmodule:: {{ fullname }}

.. --------------------------------------------------
..  Attributes
.. --------------------------------------------------

{% block attributes %}
{% if attributes %}
.. rubric:: {{ _('Module Attributes') }}

.. autosummary::
{% for item in attributes %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

.. --------------------------------------------------
..  Functions
.. --------------------------------------------------

{% block functions %}
{% if functions %}
.. rubric:: {{ _('Functions') }}

.. autosummary::
{% for item in functions %}
   {{ item }}

   .. _sphx_glr_backref_{{fullname | shorten}}.{{item}}:

   .. minigallery:: {{fullname | shorten}}.{{item}}

      :add-heading:
{%- endfor %}
{% endif %}
{% endblock %}

.. --------------------------------------------------
..  Classes
.. --------------------------------------------------

{% block classes %}
{% if classes %}
.. rubric:: {{ _('Classes') }}

.. autosummary::
{% for item in classes %}
   {{ item }}

   .. _sphx_glr_backref_{{fullname | shorten}}.{{item}}:

   .. minigallery:: {{fullname | shorten}}.{{item}}

{%- endfor %}
{% endif %}
{% endblock %}

.. --------------------------------------------------
..  Exceptions
.. --------------------------------------------------

{% block exceptions %}
{% if exceptions %}
.. rubric:: {{ _('Exceptions') }}

.. autosummary::
{% for item in exceptions %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

.. --------------------------------------------------
..  Modules
.. --------------------------------------------------

{% block modules %}
{% if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

.. --------------------------------------------------
..  Module Documentation
.. --------------------------------------------------

.. automodule:: {{ fullname }}

{% endif %}