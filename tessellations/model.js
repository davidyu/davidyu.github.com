model = (function() {
  var model = {};

  // these are constructors.
  model.squareGrid = function( gridw, gridh, DefaultTile, OutOfBoundsTile ) {
    var grid = [];

    grid.get = function( x, y ) {
      return y >= 0 && x >= 0 && x < gridw && y < gridh ? this[ x + y * gridw ] : OutOfBoundsTile;
    }

    grid.getFlat = function( i ) {
      return i >= 0 && i < gridw * gridh ? this[ i ] : OutOfBoundsTile;
    }

    grid.set = function( x, y, tile ) {
      if ( y >= 0 && x >= 0 && x < gridw && y < gridh ) {
        this[ x + y * gridw ] = tile;
      }
    }

    grid.setFlat = function( i, tile ) {
      if ( i >= 0 && i < gridw * gridh ) {
        this[ i ] = tile;
      }
    }

    for ( i = 0; i < gridw * gridh; i++ ) {
      grid.setFlat( i, DefaultTile );
    }

    return grid;
  }

  // todo...
  model.hexGrid = function() {
    console.log( "not implemented!" );
  }

  return model;
}) ();
