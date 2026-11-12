# GraphQL Apollo Federation Architecture

Apollo Federation composes multiple subgraph GraphQL services into a unified supergraph gateway. Subgraphs define entity types decorated with the @key directive specifying primary keys. The gateway compiles a query plan, resolves field dependencies across subgraphs via _entities queries, and merges the responses. The @requires and @provides directives optimize field fetching by declaring internal schema dependencies.
