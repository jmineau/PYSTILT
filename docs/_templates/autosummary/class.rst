{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

{% set hidden = [
   '__init__',
   '__weakref__',
   'DEFAULT_TARGET',
   'XYERR_PARAMS',
   'ZIERR_PARAMS',
   'construct',
   'copy',
   'dict',
   'from_orm',
   'json',
   'model_computed_fields',
   'model_config',
   'model_construct',
   'model_copy',
   'model_dump',
   'model_dump_json',
   'model_extra',
   'model_fields',
   'model_fields_set',
   'model_json_schema',
   'model_parametrized_name',
   'model_post_init',
   'model_rebuild',
   'model_validate',
   'model_validate_json',
   'model_validate_strings',
   'parse_file',
   'parse_obj',
   'parse_raw',
   'schema',
   'schema_json',
   'update_forward_refs',
   'validate',
] %}
{% set ns = namespace(methods=[], attributes=[]) %}
{% for item in methods %}
{% if item not in hidden %}
{% set ns.methods = ns.methods + [item] %}
{% endif %}
{% endfor %}
{% for item in attributes %}
{% if item not in hidden %}
{% set ns.attributes = ns.attributes + [item] %}
{% endif %}
{% endfor %}

{% if fullname.startswith('stilt.config.') %}
.. class-signature:: {{ fullname }}

Parameters
----------

.. config-model:: {{ fullname }}

{% if ns.methods %}
Methods
-------

.. autosummary::
   :toctree:

{% for item in ns.methods %}
   ~{{ objname }}.{{ item }}
{%- endfor %}
{% endif %}
{% elif fullname == 'stilt.SimID' %}
.. class-signature:: {{ fullname }}

.. class-parameters:: {{ fullname }}

Methods
-------

.. autosummary::
   :toctree:

   ~{{ objname }}.from_parts

Attributes
----------

.. autosummary::
   ~{{ objname }}.location
   ~{{ objname }}.met
   ~{{ objname }}.receptor
   ~{{ objname }}.time
{% else %}
.. class-signature:: {{ fullname }}

.. class-parameters:: {{ fullname }}

{% if ns.methods %}
Methods
-------

.. autosummary::
   :toctree:

{% for item in ns.methods %}
   ~{{ objname }}.{{ item }}
{%- endfor %}
{% endif %}
{% if ns.attributes %}
Attributes
----------

.. autosummary::
{% for item in ns.attributes %}
   ~{{ objname }}.{{ item }}
{%- endfor %}
{% endif %}
{% endif %}
