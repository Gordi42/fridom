{{ fullname }}
{{ underline }}

{{ docstring }}

{% if not fullname | is_init_py %}
.. automodule:: {{ fullname }}
   :no-members:

.. toctree::
   :maxdepth: 1
   :hidden:

   {{ fullname }}.*
{% else %}
.. automodule:: {{ fullname }}
{% endif %}

