var grid = [];
var gridRep = [];
var cellw = 120;
var cellh = 120;
var maxval = 4;
var screen = null;
var activeCells = 0;
var level = 1;
var fastdebug = false;

var hover = [];
var selected = [];

var TileType = {
  OUT_OF_BOUNDS : "$",
  EMPTY         : ".",
  REGULAR       : "*",
  CONCRETE      : "#",
  DEACTIVATED   : "`",
  LAVA          : "L",
}

// weird to do this, but I want to instantiate this object array/hash with TileType vals
// and I can't do that in the constructor like above, because JavaScript doesn't like periods
// in its initializers.
enabled = (function() {
  this[ TileType.LAVA ] = false;
  return this;
})();

var Tile = function( t, v ) {
  this.type = t;
  this.value = v;
  return this;
}

colorizer = (function() { // declare everything discreetly
  var colorizer = {};

  colorizer.scale                         = {};
  colorizer.scale[ TileType.REGULAR ]     = chroma.scale( [ '#4BF920', '#1DE5A2', '#48CC20', '#18BC49', '#0DAD6D' ] );
  colorizer.scale[ TileType.LAVA ]        = chroma.scale( [ '#AE5750', '#F96541', '#FF7939' ] );
  colorizer.scale[ TileType.DEACTIVATED ] = chroma.scale( [ '#64585A', '#64585A' ] );

  colorizer.fromValue = function( t, v ) {
    if ( v < 0 ) return chroma( 'white' ).hex();
    else         return this.scale[t]( v / 9 ).hex();
  }

  colorizer.highlightFromValue = function( t, v ) {
    if ( v < 0 ) return chroma( 'white' ).hex();
    else         return this.scale[t]( v / 9 ).brighter().hex();
  }

  colorizer.borderFromValue = function( t, v ) {
    if ( v < 0 ) return chroma( 'white' ).hex();
    else         return this.scale[t]( v / 9 ).darker().hex();
  }

  return colorizer;
})();


// translation of practical flood fill implementation as described on
// http://en.wikipedia.org/wiki/Flood_fill
floodAcquire = function( start, tile ) {
  cluster = []; // this is what we return, a cluster of vector2s pointing to cells with the same value
  marked = []; // hacky way to keep track of what we've already seen
  marked.get = function( x, y ) { return this[ x + y * gridw ] === undefined ? false : this[ x + y * gridw ]; }
  marked.set = function( x, y ) { this[ x + y * gridw ] = true; }
  Q = [];
  if ( grid.get( start.x, start.y ) != tile ) return [];
  Q.push( start );
  while( Q.length > 0 ) {
    var n = Q.shift();
    if ( grid.get( n.x, n.y ).value == tile.value &&
         grid.get( n.x, n.y ).type  == tile.type  && !marked.get( n.x, n.y ) ) {
      var w = { x: n.x, y: n.y };
      var e = { x: n.x, y: n.y };

      // move w to west until the node to the west of w is no longer id
      while( grid.get( w.x - 1, w.y ).value == tile.value &&
             grid.get( w.x - 1, w.y ).type  == tile.type ) {
        w.x--;
      }
      // move e to east until the node to the west of w is no longer id
      while( grid.get( e.x + 1, e.y ).value == tile.value &&
             grid.get( e.x + 1, e.y ).type  == tile.type ) {
        e.x++;
      }

      for ( var x = w.x; x < e.x + 1; x++ ) {
        var nn = { x: x, y: n.y };
        marked.set( nn.x, nn.y );
        cluster.push( nn );
        var north = { x: nn.x, y: nn.y - 1 };
        var south = { x: nn.x, y: nn.y + 1 };
        if ( grid.get( north.x, north.y ).value == tile.value && grid.get( north.x, north.y ).type == tile.type ) Q.push( north );
        if ( grid.get( south.x, south.y ).value == tile.value && grid.get( south.x, south.y ).type == tile.type ) Q.push( south );
      }
    }
  }
  return cluster;
}

draw = function() {
  screen.clear();

  // draw cells
  gridRep = [];
  grid.forEach( function( e, i ) {
    var cell = screen.group()
                     .transform( { x: ( i % gridw ) * cellw, y : ( Math.floor( i / gridw ) ) * cellh } );

    var rect = cell.rect( cellw, cellh )
        .attr( { fill : colorizer.fromValue( e.type, e.value ),
                 id   : 'cell' + i } )
        .mouseover(
          function() {
            if ( grid[i].type == TileType.DEACTIVATED ) return;
            target = { x: i % gridw, y: Math.floor( i / gridw ) };
            hover = floodAcquire( target, grid.get( target.x, target.y ) );
            hover.forEach( function( t ) {
              var i = t.x + t.y * gridw;
              gridRep[i].rect.attr( { fill : colorizer.highlightFromValue( grid[i].type, grid[i].value ) } );
            } );
         } )
        .mouseout(
          function() {
            target = { x: i % gridw, y: Math.floor( i / gridw ) };
            hover = floodAcquire( target, grid.get( target.x, target.y ) );
            hover.forEach( function( t ) {
              var i = t.x + t.y * gridw;
              gridRep[i].rect.attr( { fill : colorizer.fromValue( grid[i].type, grid[i].value ) } );
            } );
          }
        );

    Hammer( document.getElementById( 'cell' + i ), { preventDefault: true } ).on( "dragstart swipestart", function( e ) {
      if ( grid[i].type != TileType.DEACTIVATED ) {
        target = { x: i % gridw, y: Math.floor( i / gridw ) };
        selected = floodAcquire( target, grid.get( target.x, target.y ) );
      } else {
        selected = [];
      }
    } );

    var text = cell.plain( e.type != TileType.EMPTY ? e.value.toString() : "" )
                   .fill( { color: '#ffffff' } )
                   .transform( { x: cellw / 2, y: cellh / 2 } );

    gridRep.push( { rect : rect, text : text } );

  } );
}

update = function() {
  gridRep.forEach( function( rep, i ) {
    if ( !grid[i] ) { console.log( "tile " + i + " is undefined" ); return; }
    rep.text.plain( grid[i].type != TileType.EMPTY ? grid[i].value.toString() : "" );
    rep.rect.attr( { fill : colorizer.fromValue( grid[i].type, grid[i].value ) } )
  } );
}

advance = function() {
  level++;

  if ( fastdebug ) {
    cellw = Math.round( cellw * 0.5 );
    cellh = Math.round( cellh * 0.5 );
  } else {
    cellw = Math.round( cellw * 0.8 );
    cellh = Math.round( cellh * 0.8 );
  }

  maxval++;

  if ( level > 3 || fastdebug ) {
    enabled[ TileType.LAVA ] = true;
  }

  init();
}

prune = function( start ) {
  // see if we should delete this cell and surrounding cells
  var targets = floodAcquire( { x: start.x, y: start.y }, grid.get( start.x, start.y ) )
  if ( targets.length == grid.get( start.x, start.y ).value ) {
    if ( grid.get( start.x, start.y ).type == TileType.DEACTIVATED ) return;
    targets.forEach( function( cell ) {
      if ( grid.get( cell.x, cell.y ).type == TileType.REGULAR ) {
        grid.set( cell.x, cell.y, new Tile( TileType.EMPTY, -1 ) );
        activeCells--;
      } else if ( grid.get( cell.x, cell.y ).type == TileType.LAVA ) {
        grid.set( cell.x, cell.y, new Tile( TileType.DEACTIVATED, grid.get( cell.x, cell.y ).value ) );
      }

      console.log( activeCells );
    } );
  }
  if ( activeCells == 0 ) {
    alert( "great job, you won!" );
    advance();
  }
}

pollDrag = function( e ) {
  if ( selected == null || selected.length == 0 ) {
    console.log( "nothing selected" );
    return;
  }

  var up    = false,
      down  = false,
      left  = false,
      right = false;

  if ( Math.abs( e.gesture.deltaY ) > Math.abs( e.gesture.deltaX ) ) {
    up   = e.gesture.deltaY < 0;
    down = !up;
  } else {
    left  = e.gesture.deltaX < 0;
    right = !left;
  }

  function displace( set, direction ) {
    return set.map( function( cell ) {
      return { x: cell.x + direction.x, y: cell.y + direction.y };
    } );
  }

  function checkCollision( newset, oldset ) {
    return newset.map( function( cell, i ) {
      // if cell is out of bounds, then, collision
      // if cell is not in original set and cell is not -1 then collision
      // if cell is not in original set and cell is -1 then no collision
      // if cell is in original set then no collsion
      var cellIsOutofBounds = grid.get( cell.x, cell.y ).type == TileType.OUT_OF_BOUNDS;
      var cellInOldSet = oldset.some( function( c ) { return c.x == cell.x && c.y == cell.y } );
      var isCollision = cellIsOutofBounds || ( !cellInOldSet && grid.get( cell.x, cell.y ).type != TileType.EMPTY );
      return isCollision;
    } );
  }

  function move( from, to ) {
    // cache all the from values before clearing them
    var fromVals = from.map( function( cell ) { return grid.get( cell.x, cell.y ); } );
    from.forEach( function( cell ) { grid.set( cell.x, cell.y, new Tile( TileType.EMPTY, -1 ) ); } );
    to.forEach( function( cell, i ) { grid.set( cell.x, cell.y, fromVals[i] ); } );
  }

  if ( up ) {
    var oldset = selected.map( lib.deepCopy );
    var newset = oldset.map( lib.deepCopy );
    while( checkCollision( newset, oldset ).every( function( col ) { return col == false; } ) ) {
      oldset = newset.map( lib.deepCopy ); // oldset = newset (deep copy)
      newset = displace( oldset, { x: 0, y: -1 } );
    }
    move( selected, oldset );
    selected = oldset; // shallow copy is fine
  } else if ( down ) {
    var oldset = selected.map( lib.deepCopy );
    var newset = oldset.map( lib.deepCopy );
    while( checkCollision( newset, oldset ).every( function( col ) { return col == false; } ) ) {
      oldset = newset.map( lib.deepCopy ); // oldset = newset (deep copy)
      newset = displace( oldset, { x: 0, y: 1 } );
    }
    move( selected, oldset );
    selected = oldset; // shallow copy is fine
  }

  if ( left ) {
    var oldset = selected.map( lib.deepCopy );
    var newset = oldset.map( lib.deepCopy );
    while( checkCollision( newset, oldset ).every( function( col ) { return col == false; } ) ) {
      oldset = newset.map( lib.deepCopy );
      newset = displace( oldset, { x: -1, y: 0 } );
    }
    move( selected, oldset );
    selected = oldset; // shallow copy is fine
  } else if ( right ) {
    var oldset = selected.map( lib.deepCopy );
    var newset = oldset.map( lib.deepCopy );
    while( checkCollision( newset, oldset ).every( function( col ) { return col == false; } ) ) {
      oldset = newset.map( lib.deepCopy );
      newset = displace( oldset, { x: 1, y: 0 } );
    }
    move( selected, oldset );
    selected = oldset; // shallow copy is fine
  }

  if ( up || down || left || right ) {
    prune( selected[0] );
    update();
    draw();
  }
}

init = function() {
  if ( screen == null )
    screen = SVG( 'screen' ).size( 482, 482 );

  gridw = Math.floor( screen.width() / cellw );
  gridh = Math.floor( screen.height() / cellh );

  console.log( "new grid size is " + gridw + "x" + gridh );

  var tiles = [];

  grid = [];
  grid.get = function( x, y ) {
    return y >= 0 && x >= 0 && x < gridw && y < gridh ? this[ x + y * gridw ] : new Tile( TileType.OUT_OF_BOUNDS, -1 );
  }

  grid.set = function( x, y, tile ) {
    if ( y >= 0 && x >= 0 && x < gridw && y < gridh ) {
      grid[ x + y * gridw ] = tile;
    }
  }

  // set up grid
  for ( i = 0; i < gridw * gridh; i++ ) {
    var val = Math.round( Math.random() * ( maxval - 2 ) + 2 );
    grid.push( new Tile( TileType.REGULAR, val ) );
    if ( tiles[ val ] == null ) {
      tiles[ val ] = 0;
    }
    tiles[ val ]++;
  }

  // delete random elements
  for ( i = 0; i < gridw * gridh; i++ ) {
    if ( Math.random() > 0.4 ) {
      if ( tiles[ grid[i].value ] > grid[i].value ) {
        tiles[ grid[i].value ]--;
        grid[i] = new Tile( TileType.EMPTY, -1 );
      }
    }
  }

  // delete necessary elements
  for ( i = 2; i <= maxval; i++ ) {
    if ( tiles[i] > i ) {
      var count = tiles[i];
      for ( j = 0; j < gridw * gridh && tiles[i] > i; j++ ) {
        if ( grid[j].value == i ) {
          grid[j] = new Tile( TileType.EMPTY, -1 );
          tiles[i]--;
        }
      }
    }
  }

  // add necessary elements
  for ( i = 2; i <= maxval; i++ ) {
    if ( tiles[i] < i ) {
      var count = tiles[i];
      while ( tiles[i] < i ) {
        var randIndex = Math.round( Math.random() * ( gridw * gridh - 1 ) );
        if ( grid[ randIndex ].type == TileType.EMPTY ) {
          grid[ randIndex ] = new Tile( TileType.REGULAR, i );
          tiles[i]++;
        }
      }
    }
  }

  // count active cells
  activeCells = 0;
  for ( i = 2; i <= maxval; i++ ) {
    if ( tiles[i] ) {
      activeCells += tiles[i];
    } else {
      console.log( "bad state:" + i + " is null" );
    }
  }

  if ( enabled[ TileType.LAVA ] ) {
    tiles = [];
    for ( i = 0; i < gridw * gridh; i++ ) {
      if ( grid[i].type == TileType.EMPTY ) {
        if ( Math.random() > 0.4 ) {
          var val = Math.round( Math.random() * ( maxval - 4 ) + 2 );
          grid[i] = new Tile( TileType.LAVA, val );
          if ( tiles[ val ] == null ) {
            tiles[ val ] = 0;
          }
          tiles[ val ]++;
        }
      }
    }

    // delete random elements
    for ( i = 0; i < gridw * gridh; i++ ) {
      if ( Math.random() > 0.4 ) {
        if ( grid[i].type == TileType.LAVA && tiles[ grid[i].value ] > grid[i].value ) {
          tiles[ grid[i].value ]--;
          grid[i] = new Tile( TileType.EMPTY, -1 );
        }
      }
    }

    // delete necessary elements
    for ( i = 2; i <= maxval; i++ ) {
      if ( tiles[i] > i ) {
        var count = tiles[i];
        for ( j = 0; j < gridw * gridh && tiles[i] > i; j++ ) {
          if ( grid[j].type == TileType.LAVA && grid[j].value == i ) {
            grid[j] = new Tile( TileType.EMPTY, -1 );
            tiles[i]--;
          }
        }
      }
    }

    // add necessary elements
    for ( i = 2; i <= maxval; i++ ) {
      if ( tiles[i] < i ) {
        var count = tiles[i];
        while ( tiles[i] < i ) {
          var randIndex = Math.round( Math.random() * ( gridw * gridh - 1 ) );
          if ( grid[ randIndex ].type == TileType.EMPTY ) {
            grid[ randIndex ] = new Tile( TileType.LAVA, i );
            tiles[i]++;
          }
        }
      }
    }
  }

  draw();

  Hammer( document.getElementById( 'screen' ), { preventDefault: true } ).on( "dragend swipeend", pollDrag );
}
