{{ fullname | item_name | escape | underline}}

{% if fullname | is_init %}

Module: :py:mod:`{{ fullname }}`

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

{% block imports %}
{% set all_imports = fullname | get_imports %}
{% if all_imports %}

.. rubric:: {{ _('Imports') }}


.. autosummary::
   :toctree:
   :recursive:
   {% for item in all_imports %}
   {{ item }}
   {%- endfor %}



.. automodule:: {{ fullname }}
   :no-members:
   :no-inherited-members:
   :no-special-members:
   :no-private-members:
   :undoc-members:

{% endif %}
{% endblock %}



{% else %}
.. automodule:: {{ fullname }}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Module Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% if functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
   {% for item in functions %}
      {{ item }}

   .. _sphx_glr_backref_{{fullname}}.{{item}}:

   .. minigallery:: {{fullname}}.{{item}}

       :add-heading:
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
   {% for item in classes %}
      {{ item }}

   .. _sphx_glr_backref_{{fullname}}.{{item}}:

   .. minigallery:: {{fullname}}.{{item}}

   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   .. autosummary::
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

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

{% endif %}