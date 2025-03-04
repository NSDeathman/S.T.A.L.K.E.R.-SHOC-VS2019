/*
** $Id: ltable.h,v 2.8 2005/06/06 13:30:25 roberto Exp $
** Lua tables (hash)
** See Copyright Notice in lua.h
*/

#ifndef ltable_h
#define ltable_h

#include "lobject.h"

#define gnode(t, i) (&(t)->node[i])
#define gkey(n) (&(n)->i_key)
#define gval(n) (&(n)->i_val)
#define gnext(n) ((n)->i_key.next)

#define key2tval(n) (cast(const TValue*, gkey(n)))

LUAI_DATA const Node luaH_dummynode;

LUAI_FUNC const TValue* luaH_getnum(Table* t, int key);
LUAI_FUNC TValue* luaH_setnum(lua_State* L, Table* t, int key);
LUAI_FUNC const TValue* luaH_getstr(Table* t, TString* key);
LUAI_FUNC TValue* luaH_setstr(lua_State* L, Table* t, TString* key);
LUAI_FUNC const TValue* luaH_get(Table* t, const TValue* key);
LUAI_FUNC TValue* luaH_set(lua_State* L, Table* t, const TValue* key);
LUAI_FUNC Table* luaH_new(lua_State* L, int narray, int lnhash);
LUAI_FUNC void luaH_resizearray(lua_State* L, Table* t, int nasize);
LUAI_FUNC void luaH_free(lua_State* L, Table* t);
LUAI_FUNC int luaH_next(lua_State* L, Table* t, StkId key);
LUAI_FUNC int luaH_getn(Table* t);
LUAI_FUNC TValue* luaH_newkey(lua_State* L, Table* t, const TValue* key);

/* exported only for debugging */
LUAI_FUNC Node* luaH_mainposition(const Table* t, const TValue* key);

#endif
